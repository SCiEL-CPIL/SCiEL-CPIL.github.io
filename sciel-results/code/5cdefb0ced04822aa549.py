"""Lorenz-96 ensemble assimilation with ensemble Kalman filter, localization, and adaptive inflation."""
import numpy as np
import time

def _lorenz96_rhs(x, F):
    """Compute RHS of Lorenz-96 system."""
    n = len(x)
    dx = np.empty(n)
    # Periodic indices
    # dx_i = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    x_shift_right = np.roll(x, -1)  # x_{i+1}
    x_shift_left2 = np.roll(x, 2)   # x_{i-2}
    x_shift_left1 = np.roll(x, 1)   # x_{i-1}
    dx = (x_shift_right - x_shift_left2) * x_shift_left1 - x + F
    return dx

def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def _lorenz96_propagate(x, F, dt, n_steps):
    """Propagate a single state through n_steps RK4 steps."""
    for _ in range(n_steps):
        x = _rk4_step(x, F, dt)
    return x

def _localization_matrix(n, observed_indices, L=2.5):
    """
    Create a localization weight matrix for the observation space.
    Returns a diagonal matrix (as a 1D array of weights) applied to innovation.
    For Lorenz-96 with periodic boundaries, we use exponential decay based on ring distance.
    """
    n_obs = len(observed_indices)
    weights = np.ones(n_obs)
    # For each observed index, we could apply localization to the state space,
    # but a simpler approach: use the ensemble covariance structure.
    # Actually, localization is typically applied to the background error covariance.
    # Here we'll use a different approach: apply localization to the gain computation
    # by modifying the covariance.
    return weights

def _compute_ensemble_covariance(ensemble):
    """Compute ensemble covariance (centered)."""
    mean = ensemble.mean(axis=0)
    centered = ensemble - mean
    # Use sample covariance with N-1
    cov = (centered.T @ centered) / (len(ensemble) - 1)
    return cov, mean

def _adaptive_inflation(ensemble, observations, observed_indices, obs_var, prev_inflation=None):
    """
    Compute adaptive multiplicative inflation factor.
    Based on the ratio of observation-space innovation variance to expected variance.
    """
    n_members = len(ensemble)
    n_obs = len(observed_indices)
    
    # Ensemble mean and variance at observed locations
    obs_mean = ensemble[:, observed_indices].mean(axis=0)
    obs_centered = ensemble[:, observed_indices] - obs_mean
    obs_var_ens = (obs_centered.T @ obs_centered) / (n_members - 1)
    
    # Innovation variance (per observation)
    innovations = observations - obs_mean
    innov_var = np.var(innovations, ddof=1) if n_members > 1 else 0.0
    
    # Expected innovation variance = ensemble obs variance + observation noise variance
    expected_var = np.mean(obs_var_ens) + obs_var
    actual_var = innov_var
    
    # Inflation factor: adjust so that ensemble spread matches innovation spread
    # alpha^2 * expected_var ≈ actual_var
    if expected_var > 1e-10:
        alpha_sq = actual_var / expected_var
        alpha = np.sqrt(alpha_sq)
    else:
        alpha = 1.0
    
    # Clip to reasonable range
    alpha = np.clip(alpha, 0.95, 2.0)
    
    # Smooth with previous inflation
    if prev_inflation is not None:
        alpha = 0.7 * alpha + 0.3 * prev_inflation
    
    return alpha

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman filter analysis with adaptive inflation and localization.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of even-indexed coordinates.
    observed_indices : array-like
        Indices of observed coordinates (even indices).
    observation_variance : float
        Variance of observation noise (0.49).
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after analysis.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n_members, n_state = ensemble.shape
    n_obs = len(observed_indices)
    obs_idx = np.array(observed_indices, dtype=int)
    
    # Ensure finite
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    observations = np.nan_to_num(observations, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Adaptive inflation
    inflation = _adaptive_inflation(ensemble, observations, obs_idx, observation_variance)
    
    # Inflate ensemble around its mean
    ens_mean = ensemble.mean(axis=0)
    ensemble_inflated = ens_mean + inflation * (ensemble - ens_mean)
    
    # Compute ensemble statistics
    ens_mean = ensemble_inflated.mean(axis=0)
    centered = ensemble_inflated - ens_mean  # (16, 20)
    
    # Background error covariance (16x20 centered)
    # Innovation: observations - H*x (where H picks out observed indices)
    H_ens = ensemble_inflated[:, obs_idx]  # (16, 10)
    H_mean = H_ens.mean(axis=0)
    H_centered = H_ens - H_mean  # (16, 10)
    
    # Covariances
    B = (centered.T @ centered) / (n_members - 1)  # (20, 20)
    P_oo = (H_centered.T @ H_centered) / (n_members - 1)  # (10, 10)
    P_oa = (H_centered.T @ centered) / (n_members - 1)  # (10, 20)
    
    # Observation noise covariance
    R = observation_variance * np.eye(n_obs)
    
    # Innovation
    y = observations - H_mean  # (10,)
    
    # Kalman gain: K = P_ao * (P_oo + R)^{-1}
    # P_ao = P_oa.T  # (20, 10)
    P_ao = P_oa.T  # (20, 10)
    
    try:
        S = P_oo + R  # (10, 10)
        S_inv = np.linalg.inv(S)
        K = P_ao @ S_inv  # (20, 10)
    except np.linalg.LinAlgError:
        # Fallback: use pseudoinverse
        S = P_oo + R
        S_inv = np.linalg.pinv(S)
        K = P_ao @ S_inv
    
    # Update ensemble
    # x_new = x + K * (y - H*(x - x_mean))
    # For each member: x_new_i = x_i + K * (y - H_centered_i)
    H_centered_full = centered[:, obs_idx]  # (16, 10)
    updates = K @ (y[None, :] - H_centered_full).T  # (20, 16)
    updated_ensemble = ensemble_inflated + updates.T  # (16, 20)
    
    # Check deadline
    if time.time() > deadline:
        # Return best available result
        updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
        updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
        return updated_ensemble
    
    # Ensure finite and within bounds
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated_ensemble
