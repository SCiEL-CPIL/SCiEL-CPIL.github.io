import numpy as np
import time

def _gaspari_cohn_vec(r, L):
    """Vectorized Gaspari-Cohn."""
    x = r / L
    # Mask for x > 2
    mask2 = x > 2
    # Mask for 1 < x <= 2
    mask1 = (x > 1) & (x <= 2)
    # Mask for x <= 1
    mask0 = x <= 1
    
    res = np.zeros_like(x)
    
    # x <= 1
    x0 = x[mask0]
    res[mask0] = 0.25 * (1 - x0)**5 + 2.5 * (1 - x0)**4 - 0.5 * (1 - x0)**3 + 0.5
    
    # 1 < x <= 2
    x1 = x[mask1]
    res[mask1] = 0.5 * (1 - x1)**5 - 2.5 * (1 - x1)**4 + 2.5 * (1 - x1)**3 - (5.0/6.0) * (1 - x1)**2 + 0.5
    
    # x > 2 is already 0
    
    return res

def analyze(ensemble, observations, observed_indices, observation_variance):
    start_time = time.time()
    deadline = start_time + 110.0
    
    n_members, n_vars = ensemble.shape
    result = ensemble.copy()
    
    # Precompute localization weights for observed indices
    # Weights shape: (n_obs, n_vars)
    n_obs = len(observed_indices)
    loc_weights = np.zeros((n_obs, n_vars))
    L = 5.0
    
    for j_idx, j in enumerate(observed_indices):
        for k in range(n_vars):
            dist = min(abs(k - j), n_vars - abs(k - j))
            loc_weights[j_idx, k] = _gaspari_cohn(dist, L)
            
    # 1. Adaptive Inflation
    mean = np.mean(result, axis=0)
    obs_vars = result[:, observed_indices]
    mean_obs = np.mean(obs_vars, axis=0)
    var_obs = np.var(obs_vars, axis=0, ddof=1)
    
    innovations = observations - mean_obs
    R = observation_variance
    expected_var = var_obs + R
    
    ss_innov = np.sum(innovations**2)
    ss_expected = np.sum(expected_var)
    
    if ss_expected > 1e-10:
        ratio = ss_innov / ss_expected
        alpha = np.sqrt(max(1.0, ratio))
        alpha = np.clip(alpha, 1.0, 3.0)
        result = mean + alpha * (result - mean)
    else:
        alpha = 1.0

    # 2. Sequential EnKF Update
    for j_idx, j in enumerate(observed_indices):
        if time.time() > deadline:
            break
            
        x_j = result[:, j]
        m_j = np.mean(x_j)
        p_jj = np.var(x_j, ddof=1)
        
        d_j = observations[j_idx] - m_j
        s_j = p_jj + R
        
        if s_j < 1e-10:
            continue
            
        # Compute covariance vector: Cov(x_k, x_j) for all k
        # Centered ensemble
        centered = result - np.mean(result, axis=0)
        # Covariance is (1/N-1) * X' X
        # We need column j of the covariance matrix
        # cov[:, j] = (1/(N-1)) * centered[:, j] @ centered.T
        cov_vec = (centered[:, j] @ centered.T) / (n_members - 1)
        
        K = cov_vec / s_j
        K *= loc_weights[j_idx]
        
        result += np.outer(K, d_j)
        
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
        
    result = np.clip(result, -100.0, 100.0)
    
    return result
