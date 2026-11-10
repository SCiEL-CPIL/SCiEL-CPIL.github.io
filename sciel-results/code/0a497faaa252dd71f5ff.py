"""Combined Local ESRF analysis for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _gaspari_cohn(d, c=1.5):
    """Gaspari-Cohn correlation function for localization."""
    d = np.abs(d)
    result = np.zeros_like(d, dtype=float)
    mask1 = d <= 0.5 * c
    mask2 = (d > 0.5 * c) & (d <= 1.5 * c)
    mask3 = (d > 1.5 * c) & (d <= 2.5 * c)
    
    # d <= 0.5c
    r1 = d[mask1]
    result[mask1] = (-5.0 / 4.0) * (r1 / c) ** 5 + 5.0 * (r1 / c) ** 4 - 10.0 * (r1 / c) ** 3 + 5.0 * (r1 / c)
    # 0.5c < d <= 1.5c
    r2 = d[mask2]
    result[mask2] = (1.0 / 12.0) * (r2 / c) ** 5 - (5.0 / 4.0) * (r2 / c) ** 4 + (5.0 / 2.0) * (r2 / c) ** 3 - 5.0 * (r2 / c) ** 2 + (25.0 / 12.0) * (r2 / c) + 0.5
    # 1.5c < d <= 2.5c
    r3 = d[mask3]
    result[mask3] = (-1.0 / 16.0) * (r3 / c) ** 5 + (1.0 / 4.0) * (r3 / c) ** 4 - (15.0 / 8.0) * (r3 / c) ** 3 + (5.0 / 2.0) * (r3 / c) ** 2 - (25.0 / 12.0) * (r3 / c) + 105.0 / 48.0
    
    return result

def _local_esrf_update(local_ens, local_obs_mask, y_local, obs_var):
    """
    Perform a local Ensemble Square Root Filter (ESRF) update using an orthogonal transform.
    
    Parameters:
    local_ens: (M, L) array of local ensemble states
    local_obs_mask: (L,) boolean array indicating which local variables are observed
    y_local: (n_obs,) array of local observations
    obs_var: scalar observation variance
    
    Returns:
    updated local ensemble (M, L)
    """
    M, L = local_ens.shape
    n_obs = len(y_local)
    
    if n_obs == 0:
        return local_ens.copy()
    
    # Identify observed indices in the local window
    obs_idx_in_local = np.where(local_obs_mask)[0]
    
    # Ensemble mean
    x_mean = np.mean(local_ens, axis=0)
    
    # Anomalies
    A = local_ens - x_mean  # (M, L)
    
    # Local observation operator H: (n_obs, L)
    H = np.zeros((n_obs, L))
    for k, idx in enumerate(obs_idx_in_local):
        H[k, idx] = 1.0
    
    # Innovation
    innovation = y_local - H @ x_mean  # (n_obs,)
    
    # Ensemble covariance C = A^T A / (M-1)
    C = (A.T @ A) / (M - 1)  # (L, L)
    
    # Innovation covariance S = H C H^T + R
    HC = H @ C  # (n_obs, L)
    R = obs_var * np.eye(n_obs)
    S = HC @ H.T + R  # (n_obs, n_obs)
    
    # Compute Gain K = C H^T S^-1
    # Solve S Z^T = H C^T  => Z^T = S^-1 H C^T => Z = C H^T S^-1 = K
    HC_T = H @ C.T  # (n_obs, L)
    try:
        Z_T = np.linalg.solve(S, HC_T)
        K = Z_T.T  # (L, n_obs)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        K = C @ H.T @ S_inv
    
    # Analysis mean
    x_mean_new = x_mean + K @ innovation
    
    # Analysis covariance C_a = C - K C H^T
    # Note: C H^T is (L, n_obs). K is (L, n_obs).
    # K C H^T is (L, n_obs) @ (n_obs, L)? No.
    # K = C H^T S^-1.
    # C_a = C - C H^T S^-1 H C = C - K (H C).
    # H C is (n_obs, L). K is (L, n_obs).
    # So K @ (H C) is (L, L).
    C_a = C - K @ (H @ C)
    
    # Symmetrize
    C_a = (C_a + C_a.T) / 2.0
    
    # Ensure positive semi-definiteness for Cholesky
    # Add jitter
    jitter = 1e-8 * np.eye(L)
    C_f_jittered = C + jitter
    C_a_jittered = C_a + jitter
    
    # Compute Cholesky decompositions
    try:
        L_f = np.linalg.cholesky(C_f_jittered)  # (L, L), lower triangular
        L_a = np.linalg.cholesky(C_a_jittered)  # (L, L), lower triangular
    except np.linalg.LinAlgError:
        # Fallback to eigenvalue decomposition if Cholesky fails
        # C_f = U_f diag(s_f) U_f^T
        eigvals_f, eigvecs_f = np.linalg.eigh(C_f_jittered)
        eigvals_f = np.clip(eigvals_f, 1e-12, None)
        L_f = eigvecs_f @ np.diag(np.sqrt(eigvals_f))
        
        eigvals_a, eigvecs_a = np.linalg.eigh(C_a_jittered)
        eigvals_a = np.clip(eigvals_a, 1e-12, None)
        L_a = eigvecs_a @ np.diag(np.sqrt(eigvals_a))
    
    # Compute orthogonal transform T such that T A^T A T^T = A_a^T A_a
    # We want T C_f T^T = C_a
    # C_f = L_f L_f^T, C_a = L_a L_a^T
    # T = L_a L_f^-1 is a solution, but not orthogonal.
    # To get orthogonal T:
    # Let L_f^-1 = Q_f R_f^T (QR decomposition of L_f^-1)
    # Let L_a = Q_a R_a (QR decomposition of L_a)
    # Then T = Q_a Q_f^T is orthogonal and satisfies T C_f T^T = C_a.
    
    # Compute L_f^-1
    L_f_inv = np.linalg.inv(L_f)
    
    # QR of L_f_inv: L_f_inv = Q_f R_f
    Q_f, R_f = np.linalg.qr(L_f_inv)
    
    # QR of L_a: L_a = Q_a R_a
    Q_a, R_a = np.linalg.qr(L_a)
    
    # Transform T
    T = Q_a @ Q_f.T
    
    # Update anomalies
    A_new = T @ A
    
    # Update ensemble
    updated_ens = x_mean_new + A_new
    
    return updated_ens

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Square Root Filter (ESRF) update for Lorenz-96 assimilation.
    
    Parameters:
    ensemble: (M, N) array of forecast ensemble states
    observations: (K,) array of noisy observations for observed_indices
    observed_indices: array of indices that are observed
    observation_variance: scalar variance of observation noise
    
    Returns:
    updated ensemble with same shape (M, N)
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    M, N = ensemble.shape
    K = len(observed_indices)
    
    # Start with a copy of the ensemble
    result = ensemble.copy()
    
    # Adaptive Inflation
    if M > 1:
        spread = np.std(ensemble[:, observed_indices], axis=0)
        obs_err = np.sqrt(observation_variance)
        ratio = np.mean(spread) / obs_err if obs_err > 0 else 1.0
        
        # If spread is too small, inflate more aggressively
        if ratio < 1.0:
            inflation = 1.0 + 0.05 * (1.0 - ratio)
            inflation = np.clip(inflation, 1.01, 1.05)
        else:
            inflation = 1.01
        
        # Apply multiplicative inflation around the mean
        mean = np.mean(result, axis=0)
        result = mean + inflation * (result - mean)
    else:
        inflation = 1.01

    # Local ESRF Update
    # Window size: include the target variable and its neighbors
    window_half = 2
    window_size = 2 * window_half + 1
    
    # Map global indices to observed values for quick lookup
    obs_lookup = {}
    for i, idx in enumerate(observed_indices):
        obs_lookup[idx] = observations[i]
    
    # Iterate over each state variable i
    for i in range(N):
        if time.time() > deadline:
            break
        
        # Determine local window indices
        local_indices = [(i + j) % N for j in range(-window_half, window_half + 1)]
        
        # Identify which indices in the window are observed
        local_obs_mask = np.zeros(window_size, dtype=bool)
        local_obs_vals = np.zeros(window_size)
        
        for l_idx, g_idx in enumerate(local_indices):
            if g_idx in obs_lookup:
                local_obs_mask[l_idx] = True
                local_obs_vals[l_idx] = obs_lookup[g_idx]
        
        n_obs_local = np.sum(local_obs_mask)
        
        # If no observations in this local window, keep the state as is
        if n_obs_local == 0:
            continue
        
        # Extract local ensemble states: shape (M, window_size)
        local_ens = result[:, local_indices]
        
        # Extract local observations: shape (n_obs_local,)
        y_local = local_obs_vals[local_obs_mask]
        
        # Perform local ESRF update
        updated_local_ens = _local_esrf_update(local_ens, local_obs_mask, y_local, observation_variance)
        
        # Update result
        result[:, local_indices] = updated_local_ens
    
    # Final safety checks
    result = np.clip(result, -100.0, 100.0)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return result
# EVOLVE-BLOCK-END
