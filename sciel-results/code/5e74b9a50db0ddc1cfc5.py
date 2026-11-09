"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for Lorenz-96 partial observation assimilation.
    
    Parameters:
    ensemble: (16, 20) array of ensemble members
    observations: (10,) array of observations at even indices
    observed_indices: array of observed indices (0, 2, 4, ..., 18)
    observation_variance: scalar variance of observation noise
    
    Returns:
    Updated ensemble with same (16, 20) shape
    """
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=np.int64)
    
    n_members, n_vars = ensemble.shape
    m_obs = len(observed_indices)
    
    # Compute ensemble mean and anomalies
    ens_mean = np.mean(ensemble, axis=0)
    ens_anom = ensemble - ens_mean
    
    # Build localization matrix with periodic distance
    loc_width = 3.0
    loc_mat = np.ones((m_obs, m_obs))
    for i in range(m_obs):
        for j in range(m_obs):
            dist = abs(int(observed_indices[i]) - int(observed_indices[j]))
            dist = min(dist, n_vars - dist)
            loc_mat[i, j] = np.exp(-0.5 * (dist / loc_width) ** 2)
    
    # Compute background covariance in observation space
    H_anom = ens_anom[:, observed_indices]  # (n_members, m_obs)
    P_oo = (H_anom.T @ H_anom) / (n_members - 1)  # (m_obs, m_obs)
    
    # Apply localization
    P_oo_localized = P_oo * loc_mat
    
    # Observation covariance
    R = observation_variance * np.eye(m_obs)
    
    # Innovation
    innovation = observations - ens_mean[observed_indices]
    
    # Compute Kalman gain matrix S_inv
    S = P_oo_localized + R
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Sequential (LESRF) update
    updated_ens = ensemble.copy()
    for k in range(m_obs):
        # Compute cross-covariance between obs k and all variables
        P_xy_raw = (H_anom[:, k:k+1].T @ ens_anom) / (n_members - 1)  # (1, n_vars)
        P_xy_raw = P_xy_raw[0]  # (n_vars,)
        
        # Apply localization to cross-covariance
        P_xy = P_xy_raw * loc_mat[k, :]  # (n_vars,)
        
        # Denominator: local background variance + obs variance
        denom = P_oo_localized[k, k] + observation_variance
        
        if denom < 1e-12:
            continue
        
        # Kalman gain
        K = P_xy / denom  # (n_vars,)
        
        # Innovation for this member at obs k
        member_innov = observations[k] - updated_ens[:, observed_indices[k]]  # (n_members,)
        
        # Update all variables
        updated_ens = updated_ens + np.outer(member_innov, K)
    
    # Spread preservation: adaptive inflation
    pre_spread = np.std(ensemble, axis=0)
    post_mean = np.mean(updated_ens, axis=0)
    post_spread = np.std(updated_ens, axis=0)
    
    for j in range(n_vars):
        if post_spread[j] > 1e-8 and pre_spread[j] > 1e-8:
            target = pre_spread[j] * 0.97
            if post_spread[j] < target:
                scale = target / post_spread[j]
                scale = min(scale, 1.5)
                updated_ens[:, j] = post_mean[j] + (updated_ens[:, j] - post_mean[j]) * scale
        elif post_spread[j] < 1e-8:
            # Re-inject spread if collapsed
            updated_ens[:, j] = post_mean[j] + np.random.normal(0, 0.05, n_members)
    
    # Safety: ensure finite and bounded
    if not np.all(np.isfinite(updated_ens)):
        updated_ens = np.nan_to_num(updated_ens, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ens = np.clip(updated_ens, -100.0, 100.0)
    
    return updated_ens
# EVOLVE-BLOCK-END
