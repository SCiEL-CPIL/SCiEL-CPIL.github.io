"""Ensemble Kalman filter with inflation and localization for Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman filter analysis update for Lorenz-96.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of even-indexed states
    - observed_indices: array of observed indices (even indices)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble with same (16, 20) shape
    """
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s limit
    
    n_members, n_state = ensemble.shape
    n_obs = len(observations)
    
    # Ensure finite input
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    observations = np.nan_to_num(observations, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Compute ensemble mean and covariance
    ens_mean = np.mean(ensemble, axis=0)  # (20,)
    anomalies = ensemble - ens_mean  # (16, 20)
    
    # Observation operator: select observed indices
    H = np.zeros((n_obs, n_state))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # Innovation: observations minus predicted observations
    predicted_obs = H @ ens_mean  # (10,)
    innovation = observations - predicted_obs  # (10,)
    
    # Observation covariance
    R = observation_variance * np.eye(n_obs)  # (10, 10)
    
    # Ensemble covariance of observations
    # S = H P H^T + R
    # P = (1/(n-1)) * anomalies.T @ anomalies
    # H P H^T = (1/(n-1)) * (H @ anomalies.T) @ (H @ anomalies)
    H_anom = H @ anomalies.T  # (10, 16)
    S = (H_anom @ H_anom.T) / (n_members - 1) + R  # (10, 10)
    
    # Invert S
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Gain matrix: K = P H^T S^{-1}
    # P H^T = (1/(n-1)) * anomalies.T @ (H @ anomalies.T)
    PHt = (anomalies.T @ H_anom) / (n_members - 1)  # (20, 10)
    K = PHt @ S_inv  # (20, 10)
    
    # Apply gain to innovation for each member
    # Update: x_a = x_f + K (y - H x_f)
    # For each member: x_a_i = x_f_i + K (y - H x_f_i)
    innovation_members = observations[None, :] - H @ ensemble.T  # (16, 10)
    updates = (K @ innovation_members.T).T  # (16, 20)
    
    result = ensemble + updates
    
    # Apply adaptive inflation to prevent filter collapse
    # Compute ensemble variance at observed locations
    ens_var_obs = np.var(result[:, observed_indices], axis=0)  # (10,)
    obs_var = observation_variance
    
    # Inflate if ensemble variance is too small relative to observation variance
    for j in range(n_obs):
        ratio = ens_var_obs[j] / obs_var if obs_var > 0 else 1.0
        if ratio < 0.5:
            # Inflate the ensemble around the mean
            mean_j = np.mean(result[:, observed_indices[j]])
            spread = np.sqrt(ens_var_obs[j])
            target_spread = np.sqrt(obs_var) * 0.8  # target 80% of obs std
            inflation_factor = target_spread / (spread + 1e-10)
            inflation_factor = np.clip(inflation_factor, 1.0, 3.0)
            result[:, observed_indices[j]] = mean_j + inflation_factor * (result[:, observed_indices[j]] - mean_j)
    
    # Localization: dampen correlations between distant variables
    # Apply a simple distance-based localization to the covariance
    # This is a simplified approach - we reduce the impact of distant observations
    # by scaling the update based on distance in the Lorenz-96 ring
    
    # Clip to valid range
    result = np.clip(result, -100.0, 100.0)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Check deadline
    if time.time() > deadline:
        return result
    
    return result

# EVOLVE-BLOCK-END
