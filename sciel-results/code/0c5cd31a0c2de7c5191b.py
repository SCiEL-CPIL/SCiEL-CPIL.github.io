"""
Local Ensemble Kalman Filter (LEKF) for Lorenz-96 assimilation.
Strategy:
1. Propagate ensemble forward using RK4 with estimated F.
2. Apply localized EKF update with adaptive inflation.
3. Use Gaussian spatial localization to avoid spurious correlations.
"""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_deriv(x, F):
    """Compute dx/dt for Lorenz-96 system. x is (N, 20)."""
    x_1 = np.roll(x, 1, axis=1)
    x_m1 = np.roll(x, -1, axis=1)
    x_m2 = np.roll(x, 2, axis=1)
    return (x_1 - x_m2) * x_m1 - x + F

def _rk4_step(x, F, dt, n_steps=1):
    """Propagate ensemble using RK4 for n_steps steps of size dt."""
    for _ in range(n_steps):
        k1 = _lorenz96_deriv(x, F)
        k2 = _lorenz96_deriv(x + 0.5 * dt * k1, F)
        k3 = _lorenz96_deriv(x + 0.5 * dt * k2, F)
        k4 = _lorenz96_deriv(x + dt * k3, F)
        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x

def _localization_weights(dist, length_scale):
    """Gaussian localization weights based on distance."""
    return np.exp(-0.5 * (dist / length_scale) ** 2)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Localized Ensemble Kalman Filter update for Lorenz-96.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (16, 10) array of noisy observations (even indices)
    - observed_indices: array of observed state indices (even indices)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - Updated ensemble (16, 20)
    """
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    # Estimate F from the ensemble mean dynamics
    # Use the fact that in steady state, the mean derivative should be small
    # Estimate F from the current ensemble mean
    mean_x = np.mean(ensemble, axis=0)
    # Compute derivative at mean
    x_1 = np.roll(mean_x, 1)
    x_m1 = np.roll(mean_x, -1)
    x_m2 = np.roll(mean_x, 2)
    deriv_mean = (x_1 - x_m2) * x_m1 - mean_x
    # F estimate: in steady state, deriv should be ~0, so F ~ -deriv_mean
    # But we need a more robust estimate. Use the average of (x_i - (x_{i+1}-x_{i-2})*x_{i-1})
    # Actually, let's use a simpler approach: estimate F from the observations
    # The observed variables should be close to the truth, so use them to estimate F
    
    # Alternative: use a fixed F estimate based on typical Lorenz-96 behavior
    # F=6,8,10 are the three cases. Let's estimate from the data.
    # For each observed index, compute the implied F
    F_estimates = []
    for i in observed_indices:
        i_1 = (i + 1) % n_vars
        i_m1 = (i - 1) % n_vars
        i_m2 = (i - 2) % n_vars
        # dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
        # In steady state, dx_i/dt ≈ 0, so F ≈ x_i - (x_{i+1} - x_{i-2}) * x_{i-1}
        F_est = mean_x[i] - (mean_x[i_1] - mean_x[i_m2]) * mean_x[i_m1]
        F_estimates.append(F_est)
    F = np.mean(F_estimates)
    
    # Clamp F to reasonable range
    F = np.clip(F, 4.0, 12.0)
    
    # Propagate ensemble forward by 5 steps of dt=0.01
    dt = 0.01
    n_steps = 5
    ensemble_prop = _rk4_step(ensemble, F, dt, n_steps)
    
    # Compute background mean and anomalies
    bg_mean = np.mean(ensemble_prop, axis=0)
    anomalies = ensemble_prop - bg_mean  # (16, 20)
    
    # Compute localized background covariance at observed locations
    # For each observed index, compute the covariance with all variables
    loc_range = 2.8
    obs_cov = np.zeros((n_obs, n_vars))
    
    for k, obs_idx in enumerate(observed_indices):
        for j in range(n_vars):
            dist = min(abs(obs_idx - j), n_vars - abs(obs_idx - j))
            w = _localization_weights(dist, loc_range)
            # Local covariance: weighted average of products of anomalies
            # C[obs_idx, j] = (1/(n_members-1)) * sum_m a_m[obs_idx] * a_m[j]
            cov_val = np.mean(anomalies[:, obs_idx] * anomalies[:, j])
            obs_cov[k, j] = w * cov_val
    
    # Observation error covariance
    R = observation_variance * np.eye(n_obs)
    
    # Innovation: observations - projected background mean
    bg_obs = bg_mean[observed_indices]
    innovation = observations - bg_obs  # (16, n_obs)
    
    # Compute Kalman gain
    # S = H P H^T + R, where H is the observation operator
    # P is the background covariance
    # H P H^T = obs_cov @ obs_cov.T / loc_scale (approximately)
    # Actually, let's compute S properly
    
    # S[k1, k2] = sum_j obs_cov[k1, j] * obs_cov[k2, j] / (some normalization) + R[k1,k2]
    # This is an approximation. Let's use a simpler approach.
    
    # Compute the covariance of observed variables
    P_obs = np.zeros((n_obs, n_obs))
    for k1 in range(n_obs):
        for k2 in range(n_obs):
            dist = min(abs(observed_indices[k1] - observed_indices[k2]), 
                      n_vars - abs(observed_indices[k1] - observed_indices[k2]))
            w = _localization_weights(dist, loc_range)
            P_obs[k1, k2] = w * np.mean(anomalies[:, observed_indices[k1]] * anomalies[:, observed_indices[k2]])
    
    S = P_obs + R
    
    # Invert S
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Kalman gain: K = P H^T S^{-1}
    # P H^T has shape (n_vars, n_obs)
    PHt = np.zeros((n_vars, n_obs))
    for j in range(n_vars):
        for k in range(n_obs):
            dist = min(abs(j - observed_indices[k]), n_vars - abs(j - observed_indices[k]))
            w = _localization_weights(dist, loc_range)
            PHt[j, k] = w * np.mean(anomalies[:, j] * anomalies[:, observed_indices[k]])
    
    K = PHt @ S_inv  # (n_vars, n_obs)
    
    # Apply update
    # For each member, compute the update
    updated_ensemble = ensemble_prop.copy()
    for m in range(n_members):
        # Innovation for this member
        innov_m = observations[m] - bg_obs  # (n_obs,)
        update = K @ innov_m  # (n_vars,)
        updated_ensemble[m] += update
    
    # Adaptive inflation
    # Compute innovation norm ratio
    innovation_norm = np.mean(np.sum(innovation ** 2, axis=1))
    expected_innovation_norm = np.trace(S)
    ratio = innovation_norm / (expected_innovation_norm + 1e-10)
    
    # Inflate if ratio is too small (filter collapse)
    base_inflation = 1.001
    adaptive_inflation = base_inflation + 0.01 * max(0, 1.0 - ratio)
    adaptive_inflation = np.clip(adaptive_inflation, 1.0, 1.05)
    
    # Apply inflation around the ensemble mean
    ens_mean = np.mean(updated_ensemble, axis=0)
    for m in range(n_members):
        updated_ensemble[m] = ens_mean + adaptive_inflation * (updated_ensemble[m] - ens_mean)
    
    # Clip to valid range
    updated_ensemble = np.clip(updated_ensemble, -100, 100)
    
    # Ensure finite
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated_ensemble

# EVOLVE-BLOCK-END
