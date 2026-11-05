"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (Gain Form) with Multiplicative Inflation.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors.
    observations : ndarray, shape (10,) or (16, 10)
        Noisy observations of the observed indices.
    observed_indices : array-like
        Indices of observed coordinates (even indices: 0, 2, 4, ..., 18).
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    ensemble = np.array(ensemble, dtype=np.float64)
    observations = np.array(observations, dtype=np.float64)
    observed_indices = np.array(observed_indices)
    
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    # Normalize observations to a single vector of length n_obs
    if observations.ndim == 1:
        y = observations
    else:
        # If 2D, assume it's (n_members, n_obs) and take the mean
        y = np.mean(observations, axis=0)
        
    # 1. Compute background statistics
    x_mean = np.mean(ensemble, axis=0)
    x_anom = ensemble - x_mean  # shape (n_members, n_vars)
    
    # 2. Compute background covariance P (n_vars x n_vars)
    # P = X' X'^T / (N-1)
    # We can compute this efficiently using matrix multiplication
    P = (x_anom.T @ x_anom) / (n_members - 1)
    
    # 3. Define Observation Operator H
    # H is a sparse matrix of shape (n_obs, n_vars)
    # Instead of creating the full matrix, we can operate on slices
    # But for n_vars=20, n_obs=10, creating H is trivial and clear
    H = np.zeros((n_obs, n_vars))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
        
    # 4. Compute Observation Covariance P_oo
    # P_oo = H P H^T + R
    # H P is (n_obs, n_vars)
    HP = H @ P
    # P_oo is (n_obs, n_obs)
    P_oo = HP @ H.T
    # Add observation noise variance R = sigma^2 * I
    P_oo += observation_variance * np.eye(n_obs)
    
    # 5. Compute Kalman Gain K
    # K = P H^T P_oo^{-1}
    # P H^T is (n_vars, n_obs)
    PHT = P @ H.T
    # Invert P_oo (10x10 matrix)
    try:
        P_oo_inv = np.linalg.inv(P_oo)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular (unlikely with inflation/noise)
        P_oo_inv = np.linalg.pinv(P_oo)
        
    K = PHT @ P_oo_inv  # shape (n_vars, n_obs)
    
    # 6. Update Mean
    # Innovation = y - H x_mean
    Hx_mean = H @ x_mean
    innovation = y - Hx_mean  # shape (n_obs,)
    
    x_mean_updated = x_mean + K @ innovation  # shape (n_vars,)
    
    # 7. Update Anomalies
    # X'_a = (I - K H) X'_b
    # K H is (n_vars, n_vars)
    KH = K @ H
    I = np.eye(n_vars)
    transform_matrix = I - KH  # shape (n_vars, n_vars)
    
    x_anom_updated = x_anom.T @ transform_matrix.T  # shape (n_vars, n_members)
    x_anom_updated = x_anom_updated.T  # shape (n_members, n_vars)
    
    # 8. Reconstruct Ensemble
    updated_ensemble = x_anom_updated + x_mean_updated
    
    # 9. Multiplicative Inflation
    # Inflate anomalies around the updated mean
    # A factor of 1.02-1.05 is typical. 1.02 is conservative.
    inflation_factor = 1.02
    updated_ensemble = inflation_factor * (updated_ensemble - x_mean_updated) + x_mean_updated
    
    # 10. Ensure finite values and clip
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble
# EVOLVE-BLOCK-END
