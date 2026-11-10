"""Lorenz-96 ensemble assimilation: local EnKF with adaptive inflation and localization."""
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

    dm = d[m1]
    if len(dm) > 0:
        result[m1] = (-5/4)*(dm/c)**5 + 5*(dm/c)**4 - 10*(dm/c)**3 + 5*dm

    dm = d[m2]
    if len(dm) > 0:
        result[m2] = (1/12)*(dm/c)**5 - (5/4)*(dm/c)**4 + (5/2)*(dm/c)**3 - 5*(dm/c)**2 + (25/12)*(dm/c) + 1/2

    dm = d[m3]
    if len(dm) > 0:
        result[m3] = -(1/16)*(dm/c)**5 + (1/4)*(dm/c)**4 - (15/8)*(dm/c)**3 + (5/2)*(dm/c)**2 - (25/12)*(dm/c) + 105/48

    return result


def _periodic_distance(i, j, N):
    """Minimum distance on a ring of size N."""
    d = abs(i - j)
    return min(d, N - d)


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Kalman Filter update for Lorenz-96 assimilation.
    
    Uses a local window around each state variable, applies Gaspari-Cohn
    localization to the ensemble covariance, and performs a standard
    Kalman update. Adaptive multiplicative inflation prevents collapse.
    
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
    
    # Build observed index lookup
    obs_set = set(int(x) for x in observed_indices)
    obs_map = {int(observed_indices[k]): k for k in range(K)}
    
    # Adaptive multiplicative inflation
    if M > 2:
        obs_idx_list = list(obs_set)
        spread = np.std(ensemble[:, obs_idx_list], axis=0)
        obs_err = np.sqrt(observation_variance)
        if obs_err > 0:
            ratio = np.mean(spread) / obs_err
        else:
            ratio = 1.0
        if ratio < 1.0:
            inflation = 1.0 + 0.05 * (1.0 - ratio)
        else:
            inflation = 1.005
        inflation = float(np.clip(inflation, 1.005, 1.06))
    else:
        inflation = 1.005
    
    mean = np.mean(result, axis=0)
    result = mean + inflation * (result - mean)
    
    # Correlation length for localization
    corr_len = 4.0
    
    # Precompute localization weights for each center i
    loc_weights_by_i = []
    for i in range(N):
        dists = np.array([_periodic_distance(i, (i + j) % N, N) for j in range(-_WINDOW_HALF, _WINDOW_HALF + 1)])
        w = _gaspari_cohn(dists, corr_len)
        w[_WINDOW_HALF] = 1.0  # ensure center weight is 1
        loc_weights_by_i.append(w)
    
    # Time budget with safety margin
    time_budget = 130.0
    
    for i in range(N):
        if time.time() - t0 > time_budget:
            break
        
        # Local window indices (periodic)
        local_idx = [(i + j) % N for j in range(-_WINDOW_HALF, _WINDOW_HALF + 1)]
        n_local = len(local_idx)
        
        # Find which local positions are observed
        local_obs_pos = []
        for l, g in enumerate(local_idx):
            if g in obs_set:
                local_obs_pos.append(l)
        
        if len(local_obs_pos) == 0:
            continue
        
        n_obs_local = len(local_obs_pos)
        
        # Extract local ensemble: (M, n_local)
        local_ens = result[:, local_idx].copy()
        
        # Local observations
        y_local = np.array([observations[obs_map[local_idx[l]]] for l in local_obs_pos])
        
        # Local mean and anomalies
        local_mean = np.mean(local_ens, axis=0)
        local_anoms = local_ens - local_mean  # (M, n_local)
        
        # Apply localization to anomalies (element-wise)
        w = loc_weights_by_i[i]
        local_anoms_loc = local_anoms * w[np.newaxis, :]
        
        # Local ensemble covariance with localization
        C = (local_anoms_loc.T @ local_anoms_loc) / (M - 1)  # (n_local, n_local)
        
        # Observation operator: (n_obs_local, n_local)
        H = np.zeros((n_obs_local, n_local))
        for k, l in enumerate(local_obs_pos):
            H[k, l] = 1.0
        
        # Innovation covariance S = H C H^T + R
        HC = H @ C  # (n_obs_local, n_local)
        S = HC @ H.T + observation_variance * np.eye(n_obs_local)
        
        # Gain K = C H^T S^-1 : (n_local, n_obs_local)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        
        K_gain = C @ H.T @ S_inv  # (n_local, n_obs_local)
        
        # Update each member
        for m in range(M):
            Hx_m = H @ local_ens[m, :]  # (n_obs_local,)
            innov_m = y_local - Hx_m  # (n_obs_local,)
            local_ens_new = local_ens[m, :] + K_gain @ innov_m
            result[m, local_idx] = local_ens_new
    
    # Safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result
# EVOLVE-BLOCK-END
