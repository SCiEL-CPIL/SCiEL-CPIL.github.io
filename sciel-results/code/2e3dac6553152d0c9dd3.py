"""Local EnKF with adaptive inflation and dynamic localization for Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_deriv(x, F):
    """Compute dx/dt for Lorenz-96 system. x has shape (N, 20) or (20,)."""
    if x.ndim == 1:
        x_1 = np.roll(x, 1)
        x_m1 = np.roll(x, -1)
        x_m2 = np.roll(x, 2)
        return (x_1 - x_m2) * x_m1 - x + F
    else:
        x_1 = np.roll(x, 1, axis=1)
        x_m1 = np.roll(x, -1, axis=1)
        x_m2 = np.roll(x, 2, axis=1)
        return (x_1 - x_m2) * x_m1 - x + F


def _cyclic_distance(i, j, dim):
    """Compute cyclic distance between indices i and j in a ring of size dim."""
    d = abs(i - j)
    return min(d, dim - d)


def _localize_gain(K_local, state_idx, obs_indices, dim, decay_scale):
    """Apply exponential localization to a local gain matrix for a given state index."""
    K_loc = K_local.copy()
    for j, obs_idx in enumerate(obs_indices):
        dist = _cyclic_distance(state_idx, obs_idx, dim)
        K_loc[:, j] *= np.exp(-dist / decay_scale)
    return K_loc


def _estimate_inflation(ensemble, observations, observed_indices, observation_variance, x_mean):
    """Estimate adaptive inflation factor based on ensemble spread vs observation innovation."""
    N, dim = ensemble.shape
    n_obs = len(observed_indices)
    
    # Compute ensemble spread at observed locations
    X = ensemble - x_mean  # (N, dim)
    H = np.zeros((n_obs, dim))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # Forecast observation covariance (diagonal approximation)
    HSH_diag = np.diag(H @ (X.T @ X / (N - 1)) @ H.T)
    R_diag = np.full(n_obs, observation_variance)
    
    # Innovation variance
    y_mean = observations.mean(axis=0)
    innovation = y_mean - H @ x_mean
    innov_var = np.var(observations, axis=0)
    
    # Ratio of actual to expected innovation variance
    expected_innov_var = HSH_diag + R_diag
    ratio = np.mean(innov_var / np.maximum(expected_innov_var, 1e-10))
    
    # Target ratio ~1.0; inflate if ensemble is too narrow
    if ratio < 0.8:
        inflation = 1.0 + 0.02 * (1.0 - ratio)
    elif ratio > 1.2:
        inflation = 1.0 - 0.01 * (ratio - 1.0)
    else:
        inflation = 1.0
    
    return np.clip(inflation, 0.95, 1.05)


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Kalman Filter with adaptive inflation for Lorenz-96.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (16, 10) array of noisy observations of even-indexed coordinates
    - observed_indices: array of observed coordinate indices (even indices: 0,2,...,18)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble: (16, 20) array
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    N, dim = ensemble.shape
    n_obs = len(observed_indices)
    
    # Ensemble mean
    x_mean = np.mean(ensemble, axis=0)  # (dim,)
    
    # Centered anomalies
    X = ensemble - x_mean  # (N, dim)
    
    # Observation operator
    H = np.zeros((n_obs, dim))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # Observation mean
    y = observations.mean(axis=0)  # (n_obs,)
    innovation = y - H @ x_mean  # (n_obs,)
    
    # Observation covariance
    R = np.eye(n_obs) * observation_variance
    
    # Estimate adaptive inflation
    inflation = _estimate_inflation(ensemble, observations, observed_indices, observation_variance, x_mean)
    
    # Dynamically set localization scale based on ensemble spread
    # In Lorenz-96, correlations decay over a few grid points
    # Use a base scale of 3, modulated by spread
    spread = np.std(X, axis=0)
    mean_spread = np.mean(spread)
    # If spread is small, tighten localization; if large, loosen slightly
    decay_scale = 2.5 + 0.5 * np.clip(mean_spread / 5.0, 0.5, 1.5)
    
    # Local EnKF: compute update for each ensemble member independently
    result = np.zeros_like(ensemble)
    
    for n in range(N):
        if time.time() > deadline:
            # Fallback: return current best (mean-shifted ensemble)
            result[n] = ensemble[n] + innovation[np.newaxis, :] @ (H * np.eye(dim))  # rough fallback
            continue
        
        # Local covariance for member n
        # Use all other members for covariance estimation (leave-one-out)
        X_others = np.delete(X, n, axis=0)  # (N-1, dim)
        x_others_mean = np.mean(X_others, axis=0)
        X_others_centered = X_others - x_others_mean
        
        # Local covariance
        S_local = (X_others_centered.T @ X_others_centered) / (N - 2) if N > 2 else S_local_fallback
        
        # Forecast observation covariance
        HSH_local = H @ S_local @ H.T
        P_y_local = HSH_local + R
        
        # Kalman gain for this member
        try:
            P_y_inv = np.linalg.inv(P_y_local)
            K_local = S_local @ H.T @ P_y_inv  # (dim, n_obs)
        except np.linalg.LinAlgError:
            P_y_inv = np.linalg.pinv(P_y_local)
            K_local = S_local @ H.T @ P_y_inv
        
        # Apply localization
        K_local = _localize_gain(K_local, np.arange(dim), observed_indices, dim, decay_scale)
        
        # Innovation for this member (using member's own state)
        innovation_n = y - H @ ensemble[n]  # (n_obs,)
        
        # Update member
        result[n] = ensemble[n] + K_local @ innovation_n
    
    # Check deadline
    if time.time() > deadline:
        result = np.where(np.isfinite(result), result, 0.0)
        result = np.clip(result, -100.0, 100.0)
        return result
    
    # Apply adaptive inflation
    result_mean = np.mean(result, axis=0)
    result = result_mean + inflation * (result - result_mean)
    
    # Ensure finiteness and bounds
    result = np.where(np.isfinite(result), result, 0.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result

# EVOLVE-BLOCK-END
