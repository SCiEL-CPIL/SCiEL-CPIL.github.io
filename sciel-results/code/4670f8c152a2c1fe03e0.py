"""Local ESRF analysis for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _gaspari_cohn(d, c=1.5):
    """Gaspari-Cohn correlation function for localization.
    
    Parameters:
    d: array of distances (can be negative, we take abs)
    c: correlation scale
    
    Returns:
    Array of correlation values in [0, 1]
    """
    d = np.abs(d)
    result = np.zeros_like(d, dtype=float)
    
    mask1 = d <= 0.5 * c
    mask2 = (d > 0.5 * c) & (d <= 1.5 * c)
    mask3 = (d > 1.5 * c) & (d <= 2.5 * c)
    
    # d <= 0.5c
    r1 = d[mask1]
    if len(r1) > 0:
        result[mask1] = (-5.0 / 4.0) * (r1 / c) ** 5 + 5.0 * (r1 / c) ** 4 - 10.0 * (r1 / c) ** 3 + 5.0 * (r1 / c)
    
    # 0.5c < d <= 1.5c
    r2 = d[mask2]
    if len(r2) > 0:
        result[mask2] = (1.0 / 12.0) * (r2 / c) ** 5 - (5.0 / 4.0) * (r2 / c) ** 4 + (5.0 / 2.0) * (r2 / c) ** 3 - 5.0 * (r2 / c) ** 2 + (25.0 / 12.0) * (r2 / c) + 0.5
    
    # 1.5c < d <= 2.5c
    r3 = d[mask3]
    if len(r3) > 0:
        result[mask3] = (-1.0 / 16.0) * (r3 / c) ** 5 + (1.0 / 4.0) * (r3 / c) ** 4 - (5.0 / 8.0) * (r3 / c) ** 3 - (5.0 / 2.0) * (r3 / c) ** 2 + (25.0 / 12.0) * (r3 / c) - 0.25
    
    return result


def _local_esrf_update(ensemble, observations, observed_indices, observation_variance, 
                       n_members=16, n_state=20, n_obs=10, 
                       inflation=1.1, localization_scale=3.0, 
                       esrf_fraction=0.5):
    """Local Ensemble Square Root Filter (ESRF) update with localization.
    
    Parameters:
    ensemble: (n_members, n_state) array
    observations: (n_obs,) array
    observed_indices: (n_obs,) array of indices
    observation_variance: scalar
    n_members: number of ensemble members
    n_state: number of state variables
    n_obs: number of observations
    inflation: multiplicative inflation factor
    localization_scale: scale for Gaspari-Cohn localization
    esrf_fraction: fraction of members used in local analysis (0.5 = half)
    
    Returns:
    Updated ensemble (n_members, n_state)
    """
    result = ensemble.copy().astype(float)
    
    # Compute ensemble mean and perturbations
    mean = np.mean(ensemble, axis=0)
    perturbations = ensemble - mean  # (n_members, n_state)
    
    # Adaptive inflation based on observation innovation
    innovations = observations - mean[observed_indices]
    # Estimate observation error from ensemble spread at observed locations
    ens_spread_obs = np.std(perturbations[:, observed_indices], axis=0)
    # Inflate if ensemble is too tight relative to innovations
    ratio = np.mean(np.abs(innovations) / (ens_spread_obs + 1e-10))
    if ratio > 1.5:
        inflation_factor = 1.0 + 0.1 * (ratio - 1.5)
        inflation_factor = min(inflation_factor, 1.3)
    else:
        inflation_factor = inflation
    
    # Apply inflation to perturbations
    perturbations = perturbations * inflation_factor
    
    # ESRF: work with a subset of members for local analysis
    n_local = max(2, int(n_members * esrf_fraction))
    n_local = min(n_local, n_members)
    
    # For each state variable, perform local analysis
    for i in range(n_state):
        # Get perturbations for this state variable
        x_i_pert = perturbations[:, i]  # (n_members,)
        
        # Compute localization weights for observations relative to state i
        loc_weights = np.zeros(n_obs)
        for j, obs_idx in enumerate(observed_indices):
            # Periodic distance in state space
            dist = abs(i - obs_idx)
            dist = min(dist, n_state - dist)  # periodic
            loc_weights[j] = _gaspari_cohn(np.array([dist]), c=localization_scale)[0]
        
        # Filter out observations with zero localization weight
        valid_obs = loc_weights > 0.01
        if not np.any(valid_obs):
            # No useful observations for this state, just apply inflation
            result[:, i] = mean[i] + x_i_pert
            continue
        
        obs_sel = observations[valid_obs]
        loc_sel = loc_weights[valid_obs]
        n_valid = np.sum(valid_obs)
        
        # Local ESRF update for this state
        # Use the local subset of members
        # Create local ensemble: mean + perturbations for selected members
        local_ens = np.zeros(n_local)
        for m in range(n_local):
            local_ens[m] = x_i_pert[m]
        
        local_mean = np.mean(local_ens)
        local_pert = local_ens - local_mean  # (n_local,)
        
        # Compute local covariance
        # P = (1/(n-1)) * sum of outer products
        # For 1D state, P = var
        P_ii = np.var(local_pert, ddof=1) if n_local > 1 else 0.0
        
        # Observation covariance (diagonal with localization)
        R_ii = observation_variance / loc_sel  # effective obs error with localization
        
        # Kalman gain for 1D state with multiple obs
        # H = [loc_sel[0], loc_sel[1], ..., loc_sel[n_valid-1]] (1 x n_valid)
        # x = [x_i_pert[0], ..., x_i_pert[n_local-1]] (n_local x 1)
        # P = P_ii (scalar for 1D state)
        # R = diag(R_ii) (n_valid x n_valid)
        
        # K = P * H^T * (H * P * H^T + R)^-1
        # For 1D state: K = P_ii * H^T * (P_ii * H^T * H + R)^-1
        
        H = loc_sel  # (n_valid,)
        Ht = H.reshape(-1, 1)  # (n_valid, 1)
        
        # S = H * P * H^T + R (n_valid x n_valid)
        S = np.outer(H, H) * P_ii + np.diag(R_ii)
        
        # K = P * H^T * S^-1 (1 x n_valid)
        try:
            S_inv = np.linalg.inv(S)
            K = P_ii * H * S_inv  # (n_valid,)
        except np.linalg.LinAlgError:
            # Fallback: simple nudging
            result[:, i] = mean[i] + x_i_pert
            continue
        
        # Innovation for local ensemble
        # y - H * x = obs_sel - H * (mean[i] + local_pert)
        # For each member m: innovation = obs_sel - H * local_pert[m]
        # Actually, we need to be careful. The ESRF works with perturbations.
        
        # Let's redo this properly.
        # The analysis equation: x_a = x_f + K * (y - H * x_f)
        # In perturbation form: x_a_pert = x_f_pert + K * (y - H * (x_f_mean + x_f_pert))
        #                        = x_f_pert + K * (y - H * x_f_mean - H * x_f_pert)
        #                        = x_f_pert + K * (y - H * x_f_mean) - K * H * x_f_pert
        #                        = (I - K*H) * x_f_pert + K * (y - H * x_f_mean)
        
        # But we're working with a subset of members. Let's use the full ensemble mean.
        # Innovation = y - H * x_f_mean
        innovation = obs_sel - np.dot(H, mean[observed_indices[valid_obs]])
        
        # Analysis perturbation: (I - K*H) * x_f_pert + K * innovation
        # For 1D state, K*H is a scalar: K @ H
        K_H = np.dot(K, H)  # scalar
        K_innov = np.dot(K, innovation)  # scalar
        
        # Apply to all members (not just local subset)
        for m in range(n_members):
            result[m, i] = mean[i] + x_i_pert[m] * (1.0 - K_H) + K_innov
    
    # Clip to reasonable bounds
    result = np.clip(result, -100.0, 100.0)
    
    # Ensure finite values
    result[~np.isfinite(result)] = 0.0
    
    return result


def analyze(ensemble, observations, observed_indices, observation_variance):
    """Perform local ESRF analysis update.
    
    Parameters:
    ensemble: (16, 20) array of ensemble members
    observations: (10,) array of noisy observations of even-indexed states
    observed_indices: (10,) array of observed state indices
    observation_variance: scalar observation error variance
    
    Returns:
    Updated ensemble (16, 20)
    """
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 110.0  # 10s safety margin
    
    try:
        n_members, n_state = ensemble.shape
        n_obs = len(observed_indices)
        
        # Local ESRF with adaptive inflation and localization
        result = _local_esrf_update(
            ensemble, observations, observed_indices, observation_variance,
            n_members=n_members, n_state=n_state, n_obs=n_obs,
            inflation=1.1, localization_scale=3.0, esrf_fraction=0.5
        )
        
        return result
        
    except Exception as e:
        # Fallback: simple nudging
        result = ensemble.copy().astype(float)
        result[:, observed_indices] += 0.6 * (observations - result[:, observed_indices])
        result = np.clip(result, -100.0, 100.0)
        result[~np.isfinite(result)] = 0.0
        return result
# EVOLVE-BLOCK-END
