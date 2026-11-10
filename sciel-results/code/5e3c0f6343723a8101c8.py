"""Combined Local Stochastic ESRF analysis for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _local_esrf_update(local_ens, local_obs_mask, y_local, obs_var):
    """
    Perform a local Stochastic Ensemble Square Root Filter update.
    
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
    S = HC @ H.T + obs_var * np.eye(n_obs)  # (n_obs, n_obs)
    
    # Compute Kalman Gain K = C H^T S^{-1}
    # To avoid explicit inverse, solve S Z^T = H C^T
    try:
        HC_T = H @ C.T  # (n_obs, L)
        Z_T = np.linalg.solve(S, HC_T)  # (n_obs, L)
        K = Z_T.T  # (L, n_obs)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        K = C @ H.T @ S_inv
    
    # Deterministic update for each member
    # x_m^a = x_m + K (y - H x_m)
    updated_ens = np.zeros_like(local_ens)
    for m in range(M):
        Hx = H @ local_ens[m, :]
        updated_ens[m, :] = local_ens[m, :] + K @ (y_local - Hx)
    
    # Stochastic inflation to prevent collapse and capture non-Gaussian tails
    # This is a key structural change: instead of deterministic spread preservation,
    # we inject controlled noise to maintain diversity.
    if M > 1:
        # Compute the ratio of actual spread to expected spread at observed locations
        ens_obs = updated_ens[:, obs_idx_in_local]
        ens_obs_mean = np.mean(ens_obs, axis=0)
        ens_obs_anom = ens_obs - ens_obs_mean
        actual_spread_var = np.mean(np.var(ens_obs_anom, axis=0))
        
        # Theoretical analysis variance at observed locations should be less than obs_var
        # but we want to maintain some spread. Target spread variance ~ 0.5 * obs_var
        target_spread_var = 0.5 * obs_var
        
        if actual_spread_var > 1e-10:
            # If spread is too small, inflate
            if actual_spread_var < target_spread_var:
                inflation_factor = np.sqrt(target_spread_var / actual_spread_var)
                inflation_factor = min(inflation_factor, 1.3)
                
                # Apply multiplicative inflation around the analysis mean
                mean = np.mean(updated_ens, axis=0)
                updated_ens = mean + inflation_factor * (updated_ens - mean)
            
            # Add small stochastic perturbation to maintain diversity
            # This helps with the chaotic nature and prevents degenerate ensembles
            noise_scale = 0.01 * np.sqrt(obs_var)
            noise = np.random.randn(M, L) * noise_scale
            updated_ens += noise
    
    return updated_ens

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Stochastic Ensemble Square Root Filter (ESRF) update for Lorenz-96 assimilation.
    
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
    
    # Ensure inputs are finite
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    observations = np.nan_to_num(observations, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Start with a copy of the ensemble
    result = ensemble.copy()
    
    # Pre-compute observation lookup for quick access
    obs_lookup = {}
    for i, idx in enumerate(observed_indices):
        obs_lookup[int(idx)] = observations[i]
    
    # Local window size: include the target variable and its neighbors
    window_half = 2
    window_size = 2 * window_half + 1
    
    # Iterate over each state variable i
    for i in range(N):
        if time.time() > deadline:
            break
        
        # Determine local window indices with periodic boundary
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
        
        # Perform local stochastic ESRF update
        updated_local_ens = _local_esrf_update(local_ens, local_obs_mask, y_local, observation_variance)
        
        # Update result
        result[:, local_indices] = updated_local_ens
    
    # Final safety checks
    result = np.clip(result, -100.0, 100.0)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return result
# EVOLVE-BLOCK-END
