"""Lorenz-96 ensemble assimilation with Local ESRF and Gaspari-Cohn localization."""
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
    result[m1] = (-5/4)*(dm/c)**5 + 5*(dm/c)**4 - 10*(dm/c)**3 + 5*dm
    
    dm = d[m2]
    result[m2] = (1/12)*(dm/c)**5 - (5/4)*(dm/c)**4 + (5/2)*(dm/c)**3 - 5*(dm/c)**2 + (25/12)*(dm/c) + 1/2
    
    dm = d[m3]
    result[m3] = -(1/16)*(dm/c)**5 + (1/4)*(dm/c)**4 - (15/8)*(dm/c)**3 + (5/2)*(dm/c)**2 - (25/12)*(dm/c) + 105/48
    
    return result

def _periodic_distance(i, j, N):
    """Minimum distance on a ring of size N."""
    d = abs(i - j)
    return min(d, N - d)

def _esrf_update(local_ens, y_local, H, obs_var, M):
    """
    Perform a local Ensemble Square Root Filter update.
    
    local_ens: (M, d) local ensemble
    y_local: (n_obs,) observations
    H: (n_obs, d) observation operator
    obs_var: scalar observation variance
    M: ensemble size
    
    Returns: updated local ensemble (M, d)
    """
    d = local_ens.shape[1]
    n_obs = len(y_local)
    
    # Compute ensemble mean and anomalies
    ens_mean = np.mean(local_ens, axis=0)  # (d,)
    anoms = local_ens - ens_mean  # (M, d)
    
    # Compute ensemble covariance
    C = (anoms.T @ anoms) / (M - 1)  # (d, d)
    
    # Innovation covariance S = H C H^T + R
    HC = H @ C  # (n_obs, d)
    S = HC @ H.T + obs_var * np.eye(n_obs)  # (n_obs, n_obs)
    
    # Invert S
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Posterior covariance: C_post = C - C H^T S^{-1} H C
    K = C @ H.T @ S_inv  # (d, n_obs)
    C_post = C - K @ H @ C  # (d, d)
    
    # Posterior mean: mu_post = mu + K (y - H mu)
    innov = y_local - H @ ens_mean  # (n_obs,)
    mu_post = ens_mean + K @ innov  # (d,)
    
    # Transform anomalies to have covariance C_post
    # Eigendecomposition of C
    eigvals_C, V_C = np.linalg.eigh(C)
    eigvals_C = np.maximum(eigvals_C, 1e-10)
    
    # Eigendecomposition of C_post
    eigvals_post, V_post = np.linalg.eigh(C_post)
    eigvals_post = np.maximum(eigvals_post, 1e-10)
    
    # Compute transformation
    C_inv_sqrt = V_C @ np.diag(1.0 / np.sqrt(eigvals_C)) @ V_C.T
    C_post_sqrt = V_post @ np.diag(np.sqrt(eigvals_post)) @ V_post.T
    
    # anoms_post = anoms @ C_inv_sqrt @ C_post_sqrt
    anoms_post = anoms @ C_inv_sqrt @ C_post_sqrt  # (M, d)
    
    # Reconstruct updated ensemble
    local_ens_new = mu_post[np.newaxis, :] + anoms_post  # (M, d)
    
    return local_ens_new

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Square Root Filter (ESRF) update for Lorenz-96 assimilation.
    
    Uses direct orthogonal transformation to exactly match the updated covariance,
    preserving ensemble spread without stochastic perturbations.
    
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
            inflation = 1.0 + 0.04 * (1.0 - ratio)
        else:
            inflation = 1.005
        inflation = float(np.clip(inflation, 1.005, 1.04))
    else:
        inflation = 1.005
    
    mean = np.mean(result, axis=0)
    result = mean + inflation * (result - mean)
    
    # Correlation length for localization
    corr_len = 4.0
    
    # Precompute localization weights for each center i
    loc_weights_by_i = []
    for i in range(N):
        dists = np.array([_periodic_distance(i, (i + j) % N, N) for j in range(-3, 4)])
        w = _gaspari_cohn(dists, corr_len)
        # Ensure the center weight is 1
        w[3] = 1.0
        loc_weights_by_i.append(w)
    
    # Time budget: 140s max (safety margin from 150s)
    time_budget = 140.0
    
    for i in range(N):
        if time.time() - t0 > time_budget:
            break
        
        # Local window indices (periodic)
        local_idx = [(i + j) % N for j in range(-3, 4)]
        
        # Find which local positions are observed
        local_obs_pos = []
        for l, g in enumerate(local_idx):
            if g in obs_set:
                local_obs_pos.append(l)
        
        if len(local_obs_pos) == 0:
            continue
        
        n_obs_local = len(local_obs_pos)
        
        # Extract local ensemble: (M, 7)
        local_ens = result[:, local_idx]
        
        # Local observations
        y_local = np.array([observations[obs_map[local_idx[l]]] for l in local_obs_pos])
        
        # Apply localization to the ensemble anomalies
        local_mean = np.mean(local_ens, axis=0)
        local_anoms = local_ens - local_mean
        
        w = loc_weights_by_i[i]
        local_anoms_loc = local_anoms * w[np.newaxis, :]
        
        # Reconstruct localized ensemble
        local_ens_loc = local_mean[np.newaxis, :] + local_anoms_loc
        
        # Observation operator: (n_obs_local, 7)
        H = np.zeros((n_obs_local, 7))
        for k, l in enumerate(local_obs_pos):
            H[k, l] = 1.0
        
        # Perform ESRF update
        try:
            local_ens_new = _esrf_update(local_ens_loc, y_local, H, observation_variance, M)
        except Exception:
            # Fallback to standard EnKF if ESRF fails
            local_mean_f = np.mean(local_ens_loc, axis=0)
            local_anoms_f = local_ens_loc - local_mean_f
            C_f = (local_anoms_f.T @ local_anoms_f) / (M - 1)
            HC_f = H @ C_f
            S_f = HC_f @ H.T + observation_variance * np.eye(n_obs_local)
            try:
                S_inv_f = np.linalg.inv(S_f)
            except np.linalg.LinAlgError:
                S_inv_f = np.linalg.pinv(S_f)
            K_gain_f = C_f @ H.T @ S_inv_f
            local_ens_new = np.zeros_like(local_ens_loc)
            for m in range(M):
                Hx_m = H @ local_ens_loc[m, :]
                innov_m = y_local - Hx_m
                local_ens_new[m, :] = local_ens_loc[m, :] + K_gain_f @ innov_m
        
        # Update the result
        result[:, local_idx] = local_ens_new
    
    # Safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result
# EVOLVE-BLOCK-END
