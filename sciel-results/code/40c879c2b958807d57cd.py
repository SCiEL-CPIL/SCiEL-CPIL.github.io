"""Stochastic Ensemble Kalman Filter with adaptive inflation and localization for Lorenz-96."""
import numpy as np
import time

# Module-level state for adaptive inflation
_inflation_state = {'alpha': 1.0, 'window': 20, 'buffer': []}


def _lorenz96_step(x, F, dt=0.01, steps=5):
    """Propagate Lorenz-96 state using RK4 for given number of steps."""
    n = len(x)
    for _ in range(steps):
        def f(state):
            dx = np.zeros_like(state)
            for i in range(n):
                dx[i] = (state[(i + 1) % n] - state[(i - 2) % n]) * state[(i - 1) % n] - state[i] + F
            return dx
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter (StEnKF) analysis update.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Current ensemble of states (16 members, 20 variables).
    observations : ndarray, shape (10,)
        Noisy observations of the observed (even-indexed) variables.
    observed_indices : array_like
        Indices of observed variables (even indices: 0,2,4,...,18).
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after analysis step.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    ensemble = np.array(ensemble, dtype=np.float64)
    observations = np.array(observations, dtype=np.float64)
    observed_indices = np.array(observed_indices, dtype=int)
    observation_variance = float(observation_variance)
    
    num_members, num_vars = ensemble.shape
    num_obs = len(observed_indices)
    
    # Compute ensemble mean and anomalies
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean  # shape (16, 20)
    
    # Estimate background covariance for observed variables
    # C_oo = (1/(N-1)) * A_obs * A_obs.T
    A_obs = anomalies[:, observed_indices]  # shape (16, 10)
    C_oo = (A_obs.T @ A_obs) / (num_members - 1)  # shape (10, 10)
    
    # Add small regularization to ensure positive definiteness
    C_oo += 1e-10 * np.eye(num_obs)
    
    # Observation noise covariance
    R = observation_variance * np.eye(num_obs)
    
    # Compute Kalman gain using Cholesky decomposition
    # K = C_xo * (C_oo + R)^{-1}
    # We use the ensemble-based formulation:
    # K = A * (A.T @ (C_oo + R)^{-1})^T ... let's use the standard approach
    
    # Innovation
    innovation = observations - ens_mean[observed_indices]  # shape (10,)
    
    # Compute Kalman gain via solving linear system
    # S = C_oo + R
    S = C_oo + R
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Cross-covariance C_xo = (1/(N-1)) * A^T * A_obs
    # anomalies: (16, 20), A_obs: (16, 10)
    # C_xo should be (20, 10)
    C_xo = (anomalies.T @ A_obs) / (num_members - 1)
    
    # Kalman gain K = C_xo * S^{-1}
    K = C_xo @ S_inv  # shape (20, 10)
    
    # Apply localization to Kalman gain to remove spurious correlations
    # L96 decorrelation length is short (~2-3 units)
    state_idx = np.arange(num_vars).reshape(-1, 1)
    obs_idx = observed_indices.reshape(1, -1)
    dist = np.abs(state_idx - obs_idx)
    dist = np.minimum(dist, num_vars - dist)
    loc_weights = np.exp(-dist / 2.0)
    K = K * loc_weights
    
    # Stochastic EnKF: add random perturbations to observations
    # Generate random numbers for stochasticity
    rng = np.random.default_rng()
    # Perturb observations with zero-mean noise scaled by sqrt(R)
    # This is the key stochastic element
    obs_perturbations = rng.standard_normal(num_obs) * np.sqrt(observation_variance)
    obs_perturbed = observations + obs_perturbations
    
    # Recompute innovation with perturbed observations
    innovation_perturbed = obs_perturbed - ens_mean[observed_indices]
    
    # Analysis update for each member
    # In StEnKF, we use the perturbed observation and the standard gain
    # x_a = x_b + K * (y_perturbed - y_b)
    # But we also need to handle the full ensemble update properly
    
    # Standard EnKF update:
    # x_a_i = x_b_i + K * (y - x_b_i[obs])
    # For stochastic version, we use perturbed observations
    
    # Compute the analysis increment
    # For each member, the increment is K @ (y_perturbed - x_b_i[obs])
    # But this is per-member. Let's do it efficiently.
    
    # Background observations for each member
    bg_obs = ensemble[:, observed_indices]  # shape (16, 10)
    
    # Innovation for each member with perturbed observations
    # member_innovation = obs_perturbed - bg_obs  # shape (16, 10)
    member_innovation = obs_perturbed[None, :] - bg_obs  # shape (16, 10)
    
    # Analysis increment = member_innovation @ K.T  # shape (16, 20)
    analysis_increment = member_innovation @ K.T  # shape (16, 20)
    
    # Updated ensemble
    updated_ensemble = ensemble + analysis_increment  # shape (16, 20)
    
    # Adaptive multiplicative inflation
    # Check if ensemble spread is too small
    updated_mean = np.mean(updated_ensemble, axis=0)
    updated_anomalies = updated_ensemble - updated_mean
    
    # Compute spread ratio for observed variables
    obs_spread_bg = np.std(bg_obs, axis=0)
    obs_spread_an = np.std(updated_ensemble[:, observed_indices], axis=0)
    
    # Target spread: should be comparable to background spread
    # If analysis spread < background spread, we need inflation
    target_spread = obs_spread_bg + 1e-10
    current_spread = obs_spread_an + 1e-10
    
    # Compute inflation factor
    # We want to inflate so that spread matches target
    # alpha = sqrt(mean(target_spread / current_spread))
    spread_ratio = target_spread / current_spread
    spread_ratio = np.clip(spread_ratio, 0.5, 2.0)  # Bound the ratio
    alpha_candidate = np.sqrt(np.mean(spread_ratio))
    
    # Adapt inflation over time
    state = _inflation_state
    state['buffer'].append(alpha_candidate)
    if len(state['buffer']) > state['window']:
        state['buffer'].pop(0)
    
    # Use a smoothed inflation factor
    alpha = np.mean(state['buffer'])
    alpha = np.clip(alpha, 1.0, 1.08)  # Inflation only (no deflation), tighter cap
    
    # Apply inflation about the analysis mean
    if alpha > 1.001:
        updated_ensemble = updated_mean[None, :] + alpha * (updated_ensemble - updated_mean[None, :])
    
    # Clamp to valid range
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    # Ensure finiteness
    if not np.all(np.isfinite(updated_ensemble)):
        updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated_ensemble
