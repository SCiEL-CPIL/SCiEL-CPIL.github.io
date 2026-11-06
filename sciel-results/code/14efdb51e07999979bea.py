import numpy as np
import time

def _periodic_distance(i, j, n):
    d = abs(i - j)
    return min(d, n - d)

def _localized_enkf(ensemble, observations, observed_indices, observation_variance, loc_width):
    """
    Standard Localized Ensemble Kalman Filter.
    Uses Gaussian localization on the ring.
    """
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    x_mean = np.mean(ensemble, axis=0)
    X = ensemble - x_mean  # (n_members, n_vars)
    
    # Compute observation space covariance
    X_obs = X[:, observed_indices]  # (n_members, n_obs)
    P_oo = np.cov(X_obs, rowvar=False)  # (n_obs, n_obs)
    P_oo += observation_variance * np.eye(n_obs)
    
    # Compute cross covariance between state and observations
    # We need to apply localization to the cross covariance P_xo
    # P_xo[i, k] = Cov(X_i, X_obs_k)
    
    # Efficiently compute P_xo using localization
    # P_xo is (n_vars, n_obs)
    P_xo = np.zeros((n_vars, n_obs))
    
    # Precompute localization weights for each observed index
    # For each observed index k, we need weights w_j for all j in 0..n_vars-1
    # w_j = exp(-0.5 * (dist(j, obs_idx) / width)^2)
    
    # To speed up, we can compute the full localization matrix once if n_vars is small,
    # or just loop over observed indices. n_vars=20, n_obs=10. Looping is fine.
    
    for k_idx, obs_idx in enumerate(observed_indices):
        # Compute localization weights for this observed index
        weights = np.zeros(n_vars)
        for j in range(n_vars):
            d = _periodic_distance(j, obs_idx, n_vars)
            weights[j] = np.exp(-0.5 * (d / loc_width) ** 2)
        
        # P_xo[:, k_idx] = X^T @ X_obs[:, k_idx] * weights (scaled by 1/(N-1))
        # Actually P_xo is the covariance, so divide by (N-1)
        # X is (N, n_vars), X_obs[:, k_idx] is (N,)
        # Cov(X_j, X_obs_k) = sum_m (X_mj - mean_j)(X_mk - mean_k) / (N-1)
        # Since X is centered, mean is 0.
        # So P_xo[j, k_idx] = (X[:, j] @ X_obs[:, k_idx]) / (n_members - 1) * weights[j]
        
        cov_col = (X.T @ X_obs[:, k_idx]) / (n_members - 1)
        P_xo[:, k_idx] = cov_col * weights
    
    # Compute Kalman gain K = P_xo @ inv(P_oo)
    # K is (n_vars, n_obs)
    try:
        P_oo_inv = np.linalg.inv(P_oo)
    except np.linalg.LinAlgError:
        P_oo_inv = np.linalg.pinv(P_oo)
    
    K = P_xo @ P_oo_inv
    
    # Update ensemble
    # x_new = x_mean + K @ (observations - x_mean[obs]) + (I - K @ H) @ X
    # But H is sparse. It's easier to update each member:
    # x_m_new = x_m + K @ (observations - x_m[obs])
    
    result = ensemble.copy()
    for m in range(n_members):
        innov = observations - ensemble[m, observed_indices]
        result[m] = ensemble[m] + K @ innov
    
    # Simple adaptive inflation to prevent collapse
    # Estimate innovation variance vs spread
    # If spread is too small, inflate
    # This is a heuristic
    obs_spread_var = np.var(X_obs, axis=0)
    total_obs_spread = np.mean(obs_spread_var)
    innov_var = np.var(observations - x_mean[observed_indices])
    
    if total_obs_spread > 1e-10:
        ratio = innov_var / (total_obs_spread + observation_variance)
        if ratio > 1.5: # Spread is too small
            inflation = 1.0 + 0.1 * (ratio - 1.0)
            new_mean = np.mean(result, axis=0)
            result = new_mean + (result - new_mean) * inflation
    
    return result

def analyze(ensemble, observations, observed_indices, observation_variance):
    t_start = time.time()
    deadline = t_start + 100.0
    
    # Two-pass strategy
    # Pass 1: Narrow localization to fix local errors
    result1 = _localized_enkf(ensemble, observations, observed_indices, observation_variance, loc_width=2.5)
    
    # Check time
    if time.time() > deadline:
        return np.clip(result1, -100.0, 100.0)
    
    # Pass 2: Wider localization to smooth and improve global consistency
    result2 = _localized_enkf(result1, observations, observed_indices, observation_variance, loc_width=4.0)
    
    # Check time
    if time.time() > deadline:
        return np.clip(result2, -100.0, 100.0)
    
    # Ensure finite and within bounds
    result2 = np.nan_to_num(result2, nan=0.0, posinf=100.0, neginf=-100.0)
    result2 = np.clip(result2, -100.0, 100.0)
    
    return result2
