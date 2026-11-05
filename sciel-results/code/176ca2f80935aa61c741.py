"""Lorenz-96 assimilation using Ensemble Square Root Filter with localization and adaptive inflation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# Time deadline guard
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
    Perform an Ensemble Square Root Filter analysis update with localization and adaptive inflation.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of observed values (even indices)
    - observed_indices: array of observed index positions
    - observation_variance: scalar observation error variance
    
    Returns:
    - Updated ensemble with same shape (16, 20)
    """
    n_members, n_state = ensemble.shape
    n_obs = len(observations)
    
    # Work with a copy
    ens = ensemble.astype(np.float64).copy()
    
    # Compute ensemble mean
    ens_mean = np.mean(ens, axis=0)
    
    # Observation operator: identity for observed indices
    H = np.zeros((n_obs, n_state))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # Observation error covariance
    R = np.eye(n_obs) * observation_variance
    
    # Innovation
    innovation = observations - H @ ens_mean  # (n_obs,)
    
    # Compute sample covariance of observed variables
    obs_ens = ens[:, observed_indices]  # (n_members, n_obs)
    obs_mean = np.mean(obs_ens, axis=0)  # (n_obs,)
    obs_anoms = obs_ens - obs_mean  # (n_members, n_obs)
    
    # Sample covariance of observations
    P_oo = (obs_anoms.T @ obs_anoms) / (n_members - 1)  # (n_obs, n_obs)
    P_oo += 1e-8 * np.eye(n_obs)  # regularization
    
    # Innovation covariance S = P_oo + R
    S = P_oo + R
    S += 1e-10 * np.eye(n_obs)  # additional regularization
    
    # Compute Kalman gain for observed variables: K_oo = P_oo S^{-1}
    try:
        S_inv = np.linalg.inv(S)
        K_oo = P_oo @ S_inv  # (n_obs, n_obs)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        K_oo = P_oo @ S_inv
    
    # For unobserved variables, compute cross-covariance with observed variables
    unobs_indices = [i for i in range(n_state) if i not in observed_indices]
    
    # We'll update all variables using the ensemble regression approach
    # For each unobserved variable, regress it onto the observed variables
    
    # Compute cross-covariance between all state variables and observed variables
    all_anoms = ens - ens_mean  # (n_members, n_state)
    P_xo = (all_anoms.T @ obs_anoms) / (n_members - 1)  # (n_state, n_obs)
    
    # Kalman gain for all state variables: K = P_xo S^{-1}
    K = P_xo @ S_inv  # (n_state, n_obs)
    
    # Apply localization to the Kalman gain
    # Use a Gaussian decay based on circular distance
    decay_length = 2.0
    localized_K = np.zeros_like(K)
    for j in range(n_obs):
        obs_idx = observed_indices[j]
        for i in range(n_state):
            d = abs(i - obs_idx)
            d = min(d, n_state - d)
            w = np.exp(-d ** 2 / (2.0 * decay_length ** 2))
            localized_K[i, j] = K[i, j] * w
    
    # Ensemble update using square root form
    # x_m^a = x_m^f + K_loc @ (y - H x_m^f)
    Hx = ens[:, observed_indices]  # (n_members, n_obs)
    member_innovations = observations[None, :] - Hx  # (n_members, n_obs)
    
    # Update: ens_a = ens_f + localized_K @ member_innovations^T
    update = member_innovations @ localized_K.T  # (n_members, n_state)
    ens = ens + update
    
    # Adaptive inflation: match the ensemble spread to the observed innovation variance
    ens_mean_new = np.mean(ens, axis=0)
    anomalies_new = ens - ens_mean_new
    
    # Compute the expected innovation variance from the updated ensemble
    obs_ens_new = ens[:, observed_indices]
    obs_mean_new = np.mean(obs_ens_new, axis=0)
    obs_anoms_new = obs_ens_new - obs_mean_new
    P_oo_new = (obs_anoms_new.T @ obs_anoms_new) / (n_members - 1)
    
    # Expected innovation variance
    S_expected = P_oo_new + R
    
    # Actual innovation variance (diagonal elements)
    S_expected_diag = np.diag(S_expected)
    innov_sq = innovation ** 2
    
    # Compute inflation factor based on ratio of observed to expected innovation variance
    ratios = []
    for j in range(n_obs):
        exp_var = S_expected_diag[j]
        act_var = innov_sq[j] + observation_variance
        if exp_var > 1e-10 and act_var > 1e-10:
            ratios.append(act_var / exp_var)
    
    if len(ratios) > 0:
        # Geometric mean of ratios
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
