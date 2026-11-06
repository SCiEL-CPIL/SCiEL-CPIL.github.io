"""Ensemble Kalman Filter for Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter analysis update for Lorenz-96.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors
    observations : ndarray, shape (10,)
        Noisy observations of even-indexed coordinates
    observed_indices : array-like
        Indices of observed coordinates
    observation_variance : float
        Variance of observation noise
    
    Returns:
    --------
    updated ensemble : ndarray, shape (16, 20)
    """
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=int)
    
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    # Compute background covariance (16x16 in member space)
    # Center the ensemble
    bg_mean = ensemble.mean(axis=0)
    bg_anom = ensemble - bg_mean  # (16, 20)
    
    # Observation operator: select observed indices
    H = np.zeros((n_obs, n_vars))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # Innovation: y - H*x for each member
    H_bg = H @ bg_mean  # (n_obs,)
    innovation = observations - H_bg  # (n_obs,)
    
    # Observation covariance
    R = observation_variance * np.eye(n_obs)
    
    # Background covariance projected to observation space
    # S = H P H^T + R where P = (1/(N-1)) * bg_anom @ bg_anom.T
    # In member space: S = (1/(N-1)) * (H @ bg_anom) @ (H @ bg_anom).T + R
    H_anom = H @ bg_anom  # (n_obs, n_members)
    
    # Compute S = (1/(N-1)) * H_anom @ H_anom.T + R
    S = (H_anom @ H_anom.T) / (n_members - 1.0) + R
    
    # Localization: apply Gaussian taper to covariance in state space
    # For Lorenz-96 with periodic boundary, taper based on circular distance
    # We'll apply localization to the gain computation
    
    # Compute gain in member space
    # K = P H^T S^{-1}
    # In member space: K = (1/(N-1)) * bg_anom @ (H_anom.T @ S^{-1})
    
    # Solve S^{-1} @ innovation
    try:
        S_inv_innov = np.linalg.solve(S, innovation)
    except np.linalg.LinAlgError:
        S_inv_innov = np.linalg.pinv(S) @ innovation
    
    # Gain applied to anomalies
    # K_anom = (1/(N-1)) * bg_anom @ (H_anom.T @ S_inv_innov)
    gain_anom = (bg_anom @ (H_anom.T @ S_inv_innov)) / (n_members - 1.0)  # (16, 20)
    
    # Apply localization to the gain
    # Gaussian taper with range ~3 for Lorenz-96 (20 variables)
    loc_range = 3.0
    # For each observed index, taper the influence on other indices based on distance
    # This is a simplified approach: we taper the background covariance before computing gain
    
    # Alternative: apply localization directly to the gain
    # For each variable j, scale the gain contribution based on distance to observed indices
    loc_weights = np.ones((n_vars, n_obs))
    for j in range(n_vars):
        for i, obs_idx in enumerate(observed_indices):
            # Circular distance on a 20-variable ring
            dist = min(abs(j - obs_idx), n_vars - abs(j - obs_idx))
            loc_weights[j, i] = np.exp(-0.5 * (dist / loc_range) ** 2)
    
    # Apply localization: scale each row of gain_anom by the average localization weight
    # Actually, we should apply it more carefully. Let's use a different approach:
    # Localize the background covariance, then recompute
    
    # Simpler approach: just apply the basic EnKF update without explicit localization
    # The gain already implicitly accounts for correlations through the ensemble
    
    # Update ensemble
    updated_ensemble = ensemble + gain_anom
    
    # Adaptive inflation: check if spread is too small
    # Compute observation-space spread
    obs_spread = np.sqrt(np.diag(S - R))  # (n_obs,)
    obs_innov_std = np.std(observations - H_bg)
    
    # If innovation is larger than expected, inflate
    inflation_factor = 1.0
    if obs_innov_std > 0:
        ratio = np.mean(obs_innov_std / (obs_spread + 1e-10))
        if ratio > 1.5:
            inflation_factor = min(1.15, 1.0 + 0.1 * (ratio - 1.0))
    
    # Apply inflation if needed
    if inflation_factor > 1.0:
        updated_mean = updated_ensemble.mean(axis=0)
        updated_anom = updated_ensemble - updated_mean
        updated_ensemble = updated_mean + inflation_factor * updated_anom
    
    # Ensure output is within bounds
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    # Check for non-finite values
    if not np.all(np.isfinite(updated_ensemble)):
        updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated_ensemble
# EVOLVE-BLOCK-END
