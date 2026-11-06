import numpy as np
import time

# EVOLVE-BLOCK-START

_start_time = None
_DEADLINE_FRACTION = 0.85  # Use 85% of the 150s budget for safety

def _periodic_distance(i, j, n):
    """Periodic distance on a ring of length n."""
    d = abs(i - j)
    return min(d, n - d)

def _gaussian_localization(dist, L):
    """Gaussian localization function."""
    return np.exp(-0.5 * (dist / L) ** 2)

def _compute_localization_matrix(n_vars, observed_indices, loc_width):
    """Compute (n_vars, n_obs) localization weight matrix."""
    n_obs = len(observed_indices)
    loc = np.ones((n_vars, n_obs))
    for m_idx, obs_i in enumerate(observed_indices):
        for d in range(n_vars):
            dist = _periodic_distance(d, obs_i, n_vars)
            loc[d, m_idx] = _gaussian_localization(dist, loc_width)
    return loc

def _localized_enkf(ensemble, observations, observed_indices, observation_variance, loc_width):
    """
    Localized Ensemble Kalman Filter with adaptive inflation.
    Uses Gaussian localization on the ring.
    """
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    # Ensure obs is 1D
    obs_arr = np.asarray(observations)
    if obs_arr.ndim == 2:
        obs_vec = obs_arr[0]
    else:
        obs_vec = obs_arr
    
    x_mean = np.mean(ensemble, axis=0)
    X = ensemble - x_mean  # (n_members, n_vars)
    
    # Observation space anomalies
    X_obs = X[:, observed_indices]  # (n_members, n_obs)
    
    # Compute observation space covariance P_oo
    P_oo = (X_obs.T @ X_obs) / (n_members - 1)
    P_oo += observation_variance * np.eye(n_obs)
    
    # Compute localization weights
    loc_weights = _compute_localization_matrix(n_vars, observed_indices, loc_width)
    
    # Compute cross covariance P_xo with localization
    # P_xo[j, k] = Cov(X_j, X_obs_k) * loc_weights[j, k]
    P_xo = (X.T @ X_obs) / (n_members - 1)  # (n_vars, n_obs)
    P_xo_loc = P_xo * loc_weights
    
    # Adaptive inflation based on innovation spread
    innovation = obs_vec - x_mean[observed_indices]
    innov_var = np.var(innovation)
    obs_spread_var = np.var(X_obs, axis=0)
    total_obs_spread = np.mean(obs_spread_var)
    
    inflation_factor = 1.0
    if total_obs_spread > 1e-10:
        ratio = innov_var / (total_obs_spread + observation_variance)
        if ratio > 1.2:
            inflation_factor = np.clip(1.0 + 0.2 * (ratio - 1.0), 1.0, 1.3)
        elif ratio < 0.8:
            inflation_factor = np.clip(1.0 + 0.1 * (ratio - 1.0), 0.9, 1.0)
    
    # Apply inflation to anomalies
    X_inflated = X * inflation_factor
    X_obs_inflated = X_obs * inflation_factor
    
    # Recompute P_xo with inflated anomalies
    P_xo_loc_inflated = (X_inflated.T @ X_obs_inflated) / (n_members - 1) * loc_weights
    
    # Compute Kalman gain K = P_xo @ inv(P_oo)
    # Use Cholesky for stability
    try:
        L_chol = np.linalg.cholesky(P_oo)
        # Solve L_chol * tmp = P_xo_loc_inflated.T
        tmp = np.linalg.solve(L_chol, P_xo_loc_inflated.T)
        # K = (L_chol^{-T} tmp)^T
        K = np.linalg.solve(L_chol.T, tmp).T  # (n_vars, n_obs)
    except np.linalg.LinAlgError:
        P_oo_inv = np.linalg.pinv(P_oo)
        K = P_xo_loc_inflated @ P_oo_inv
    
    # Update ensemble
    # x_new_m = x_m + K @ (observations - x_m[obs])
    result = ensemble.copy()
    K_innov = K @ innovation  # (n_vars,)
    
    # Vectorized update
    Hx_anom_K = X_obs_inflated @ K.T  # (n_members, n_vars)
    result = ensemble + K_innov[None, :] - Hx_anom_K
    
    # Re-center to preserve mean consistency
    new_mean = np.mean(result, axis=0)
    expected_mean = x_mean + K_innov
    result = result + (expected_mean - new_mean)
    
    # Safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update using two-pass Localized EnKF with adaptive inflation.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > 150 * _DEADLINE_FRACTION:
        return np.clip(ensemble, -100.0, 100.0)
    
    # Two-pass strategy
    # Pass 1: Narrow localization to fix local errors
    result1 = _localized_enkf(ensemble, observations, observed_indices, observation_variance, loc_width=2.0)
    
    # Check time
    if time.time() - _start_time > 150 * _DEADLINE_FRACTION:
        return np.clip(result1, -100.0, 100.0)
    
    # Pass 2: Wider localization to smooth and improve global consistency
    result2 = _localized_enkf(result1, observations, observed_indices, observation_variance, loc_width=3.5)
    
    # Check time
    if time.time() - _start_time > 150 * _DEADLINE_FRACTION:
        return np.clip(result2, -100.0, 100.0)
    
    # Final safety checks
    result2 = np.nan_to_num(result2, nan=0.0, posinf=100.0, neginf=-100.0)
    result2 = np.clip(result2, -100.0, 100.0)
    
    return result2

# EVOLVE-BLOCK-END
