"""Lorenz-96 ensemble assimilation with Local EnKF and Gaspari-Cohn localization."""
import numpy as np
import time

# EVOLVE-BLOCK-START
_N = 20
_WINDOW_HALF = 3  # window of 7 variables (i-3 to i+3)

def _gaspari_cohn(d, c):
    """Gaspari-Cohn C2 correlation function."""
    d = np.abs(d)
    result = np.zeros_like(d, dtype=float)
    m1 = d <= 0.5 * c
    m2 = (d > 0.5 * c) & (d <= 1.5 * c)
    m3 = (d > 1.5 * c) & (d <= 2.5 * c)
    
    if np.any(m1):
        dm = d[m1]
        result[m1] = (-5/4)*(dm/c)**5 + 5*(dm/c)**4 - 10*(dm/c)**3 + 5*dm
        
    if np.any(m2):
        dm = d[m2]
        result[m2] = (1/12)*(dm/c)**5 - (5/4)*(dm/c)**4 + (5/2)*(dm/c)**3 - 5*(dm/c)**2 + (25/12)*(dm/c) + 1/2
        
    if np.any(m3):
        dm = d[m3]
        result[m3] = -(1/16)*(dm/c)**5 + (1/4)*(dm/c)**4 - (15/8)*(dm/c)**3 + (5/2)*(dm/c)**2 - (25/12)*(dm/c) + 105/48
        
    return result

def _periodic_distance(i, j, N):
    """Minimum distance on a ring of size N."""
    d = abs(i - j)
    return min(d, N - d)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Kalman Filter update for Lorenz-96 assimilation.
    
    Parameters:
    ensemble: (M, N) array of forecast ensemble states
    observations: (K,) array of noisy observations
    observed_indices: array of observed state indices
    observation_variance: scalar observation noise variance
    
    Returns:
    Updated ensemble (M, N)
    """
    t0 = time.time()
    M, N = ensemble.shape
    K = len(observed_indices)
    result = ensemble.astype(float).copy()
    
    # Build observed index set for fast lookup
    obs_set = set(int(x) for x in observed_indices)
    obs_map = {int(observed_indices[k]): k for k in range(K)}
    
    # Adaptive multiplicative inflation
    if M > 2:
        spread = np.std(ensemble[:, list(obs_set)], axis=0)
        obs_err = np.sqrt(observation_variance)
        ratio = np.mean(spread) / obs_err if obs_err > 0 else 1.0
        # Inflation: stronger when under-dispersive
        if ratio < 1.0:
            inflation = 1.0 + 0.05 * (1.0 - ratio)
        else:
            inflation = 1.005
        inflation = float(np.clip(inflation, 1.005, 1.05))
    else:
        inflation = 1.005
    
    mean = np.mean(result, axis=0)
    result = mean + inflation * (result - mean)
    
    # Correlation length for localization
    # Lorenz-96 coupling is local. c=4.0 is a standard choice.
    corr_len = 4.0
    
    # Precompute localization weights for each center i
    # For each i, window is i-WINDOW_HALF .. i+WINDOW_HALF
    loc_weights_by_i = []
    for i in range(N):
        dists = np.array([_periodic_distance(i, (i + j) % N, N) for j in range(-_WINDOW_HALF, _WINDOW_HALF + 1)])
        w = _gaspari_cohn(dists, corr_len)
        # Ensure the center weight is exactly 1
        w[_WINDOW_HALF] = 1.0
        loc_weights_by_i.append(w)
    
    # Time budget: 140s max (safety margin from 150s)
    time_budget = 140.0
    
    for i in range(N):
        if time.time() - t0 > time_budget:
            break
        
        # Local window indices (periodic)
        local_idx = [(i + j) % N for j in range(-_WINDOW_HALF, _WINDOW_HALF + 1)]
        
        # Find which local positions are observed
        local_obs_pos = []
        for l, g in enumerate(local_idx):
            if g in obs_set:
                local_obs_pos.append(l)
        
        if len(local_obs_pos) == 0:
            continue
        
        n_obs_local = len(local_obs_pos)
        
        # Extract local ensemble: (M, local_N)
        local_ens = result[:, local_idx].copy()
        
        # Local observations
        y_local = np.array([observations[obs_map[local_idx[l]]] for l in local_obs_pos])
        
        # Local mean and anomalies
        local_mean = np.mean(local_ens, axis=0)
        local_anoms = local_ens - local_mean  # (M, local_N)
        
        # Apply localization to anomalies (element-wise)
        w = loc_weights_by_i[i]
        local_anoms_loc = local_anoms * w[np.newaxis, :]
        
        # Local ensemble covariance with localization
        # C = (1/(M-1)) * X^T X
        C = (local_anoms_loc.T @ local_anoms_loc) / (M - 1)  # (local_N, local_N)
        
        # Observation operator: (n_obs_local, local_N)
        H = np.zeros((n_obs_local, len(local_idx)))
        for k, l in enumerate(local_obs_pos):
            H[k, l] = 1.0
        
        # Innovation covariance S = H C H^T + R
        HC = H @ C  # (n_obs_local, local_N)
        S = HC @ H.T + observation_variance * np.eye(n_obs_local)
        
        # Gain K = C H^T S^-1 : (local_N, n_obs_local)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        
        K_gain = C @ H.T @ S_inv  # (local_N, n_obs_local)
        
        # Update each member
        # x_post = x_prior + K (y - H x_prior)
        for m in range(M):
            # Innovation for member m
            Hx_m = H @ local_ens[m, :]  # (n_obs_local,)
            innov_m = y_local - Hx_m  # (n_obs_local,)
            
            # Local update
            local_ens_new = local_ens[m, :] + K_gain @ innov_m
            
            # Assign back to global result
            result[m, local_idx] = local_ens_new
    
    # Safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result
# EVOLVE-BLOCK-END
