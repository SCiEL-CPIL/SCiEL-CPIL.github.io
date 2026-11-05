"""Lorenz-96 assimilation using a localized Ensemble Kalman Filter with per-component adaptive inflation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_DEADLINE = 145.0
_t0 = None

def _check_deadline():
    """Return True if we should stop to avoid timeout."""
    global _t0
    if _t0 is None:
        _t0 = time.time()
    return (time.time() - _t0) > _DEADLINE


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform a localized Ensemble Kalman Filter analysis update with per-component adaptive inflation.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of observed values (even indices)
    - observed_indices: array of observed index positions
    - observation_variance: scalar observation error variance
    
    Returns:
    - Updated ensemble with same shape (16, 20)
    """
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    n_members, n_state = ensemble.shape
    n_obs = len(observations)
    
    # Work with a copy
    ens = ensemble.astype(np.float64).copy()
    
    # Compute ensemble mean
    ens_mean = np.mean(ens, axis=0)
    
    # Get observed state components from ensemble
    ens_obs = ens[:, observed_indices]  # (n_members, n_obs)
    ens_obs_mean = np.mean(ens_obs, axis=0)  # (n_obs,)
    
    # Compute innovation
    innovation = observations - ens_obs_mean  # (n_obs,)
    
    # Compute P_oo directly from observed sub-ensemble
    obs_anoms = ens_obs - ens_obs_mean  # (n_members, n_obs)
    P_oo = (obs_anoms.T @ obs_anoms) / (n_members - 1)  # (n_obs, n_obs)
    
    # Observation error covariance
    R = np.eye(n_obs) * observation_variance
    
    # Innovation covariance S = P_oo + R
    S = P_oo + R
    S_reg = S + 1e-8 * np.eye(n_obs)
    
    # Compute cross-covariance P_xo between all state variables and observed variables
    all_anoms = ens - ens_mean  # (n_members, n_state)
    P_xo = (all_anoms.T @ obs_anoms) / (n_members - 1)  # (n_state, n_obs)
    
    # Kalman gain K = P_xo S^{-1}
    try:
        K = np.linalg.solve(S_reg, P_xo.T).T  # (n_state, n_obs)
    except np.linalg.LinAlgError:
        K = P_xo @ np.linalg.pinv(S_reg)
    
    # Apply Gaussian localization to the Kalman gain
    decay_length = 2.5
    localized_K = np.zeros_like(K)
    for j in range(n_obs):
        obs_idx = observed_indices[j]
        for i in range(n_state):
            d = abs(i - obs_idx)
            d = min(d, n_state - d)
            w = np.exp(-d ** 2 / (2.0 * decay_length ** 2))
            localized_K[i, j] = K[i, j] * w
    
    # Check deadline before expensive update
    if _check_deadline():
        return ens
    
    # Ensemble update: x_m^a = x_m^f + K_loc @ (y - H x_m^f)
    member_innovations = observations[None, :] - ens_obs  # (n_members, n_obs)
    update = member_innovations @ localized_K.T  # (n_members, n_state)
    ens = ens + update
    
    # Per-component adaptive inflation
    ens_mean_new = np.mean(ens, axis=0)
    
    # Recompute P_oo_new from updated ensemble
    ens_obs_new = ens[:, observed_indices]
    ens_obs_mean_new = np.mean(ens_obs_new, axis=0)
    obs_anoms_new = ens_obs_new - ens_obs_mean_new
    P_oo_new = (obs_anoms_new.T @ obs_anoms_new) / (n_members - 1)
    
    # Expected innovation variance per component
    S_expected_diag = np.diag(P_oo_new) + observation_variance  # (n_obs,)
    
    # Actual innovation variance per component
    innov_sq = innovation ** 2  # (n_obs,)
    act_var = innov_sq + observation_variance  # (n_obs,)
    
    # Compute per-component ratio and use geometric mean
    valid_mask = (S_expected_diag > 1e-10) & (act_var > 1e-10)
    if np.any(valid_mask):
        ratios = act_var[valid_mask] / S_expected_diag[valid_mask]
        log_ratios = np.log(np.clip(ratios, 1e-6, 1e6))
        geo_mean_ratio = np.exp(np.mean(log_ratios))
        inflation = np.sqrt(np.clip(geo_mean_ratio, 0.95, 1.25))
    else:
        inflation = 1.0
    
    # Apply inflation around the new mean
    ens = ens_mean_new + inflation * (ens - ens_mean_new)
    
    # Clip to valid range
    ens = np.clip(ens, -100.0, 100.0)
    
    # Ensure finiteness
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ens

# EVOLVE-BLOCK-END
