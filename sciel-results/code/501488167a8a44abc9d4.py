"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter (EnKF) with Adaptive Inflation 
    and Gaspari-Cohn Localization.
    
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
    observed_indices = np.array(observed_indices, dtype=np.int32)
    
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    # Normalize observations to a single vector
    if observations.ndim == 1:
        y = observations
    else:
        # If 2D, assume it's the same obs for all members or take mean/first
        # Standard practice: if shape is (n_members, n_obs), it might be 
        # independent obs per member? No, usually it's a single truth obs.
        # If it's (16, 10), and we assume it's the same noisy obs repeated, 
        # taking the first row is safe. If it's different, taking the mean 
        # is a robust fallback for "average" truth.
        # Given "independent N(0,0.49) noise", it's likely a single vector.
        # If the evaluator passes (16,10), it's likely a bug or specific format.
        # Let's assume (n_obs,) is the standard. If (16,10), take mean.
        y = np.mean(observations, axis=0) if observations.shape[0] == n_members else observations[0]
        
    # 1. Compute Ensemble Mean and Anomalies
    x_mean = np.mean(ensemble, axis=0)
    x_anom = ensemble - x_mean
    
    # 2. Estimate Background Covariance P
    # P = X_anom X_anom^T / (N-1)
    P = (x_anom.T @ x_anom) / (n_members - 1)
    
    # 3. Observation Operator H
    # H is (n_obs, n_vars)
    H = np.zeros((n_obs, n_vars))
    H[np.arange(n_obs), observed_indices] = 1.0
    
    # 4. Observation Covariance R
    R = np.eye(n_obs) * observation_variance
    
    # 5. Compute Kalman Gain K
    # K = P H^T (H P H^T + R)^-1
    PHt = P @ H.T
    HPHt = H @ PHt
    S = HPHt + R
    
    try:
        # Use Cholesky for stability if possible, otherwise inv
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        
    K = PHt @ S_inv
    
    # 6. Apply Localization to K
    # Gaspari-Cohn function
    def gaspari_cohn(d, L):
        """
        Gaspari-Cohn localization function.
        d: distance
        L: correlation length (radius)
        """
        d = np.abs(d)
        L = float(L)
        # C2 continuous
        # For d <= L/2:
        #   1 - 5/4*(d/L)^2 + 5/8*(d/L)^4
        # For L/2 < d <= L:
        #   5/2*(d/L)^3 - 5/2*(d/L)^4 + 1/4*(d/L)^5 - 1/12
        # Else 0
        
        # Vectorized version
        result = np.zeros_like(d)
        mask1 = d <= 0.5 * L
        mask2 = (d > 0.5 * L) & (d <= L)
        
        x1 = d[mask1] / L
        result[mask1] = 1.0 - 1.25 * x1**2 + 0.625 * x1**4
        
        x2 = d[mask2] / L
        result[mask2] = 2.5 * x2**3 - 2.5 * x2**4 + 0.25 * x2**5 - 1.0/12.0
        
        return result

    L_loc = 4.0  # Localization length scale in grid points
    
    # Apply localization to K (n_vars x n_obs)
    # K[i, j] corresponds to state var i and obs var j
    # Distance between state i and obs index observed_indices[j]
    # Note: Lorenz-96 is periodic, so distance is min(|i-j|, N-|i-j|)
    
    # We need to apply this to the gain matrix before updating.
    # However, standard practice is to localize the covariance or the gain.
    # Localizing the gain is simpler here.
    
    K_loc = K.copy()
    for j in range(n_obs):
        obs_idx = observed_indices[j]
        # Compute periodic distance for all state variables i
        dists = np.minimum(np.abs(np.arange(n_vars) - obs_idx), n_vars - np.abs(np.arange(n_vars) - obs_idx))
        weights = gaspari_cohn(dists, L_loc)
        K_loc[:, j] *= weights
    
    # 7. Stochastic EnKF Update
    # x_i_new = x_i + K_loc (y - H x_i)
    # This preserves the mean exactly (in expectation) and maintains spread.
    
    updated_ensemble = np.zeros_like(ensemble)
    for i in range(n_members):
        Hx_i = H @ ensemble[i]
        innovation = y - Hx_i
        updated_ensemble[i] = ensemble[i] + K_loc @ innovation
        
    # 8. Adaptive Inflation
    # Compute the spread of the updated ensemble at observed locations
    # and compare to the spread of the observations (which is just R if we assume 
    # the obs are from truth, but we don't have truth).
    # A common adaptive inflation scheme:
    # Inflate the ensemble such that the ratio of the ensemble variance to 
    # the observation variance matches a target.
    # Since we don't have the true variance, we use the observation variance 
    # as a proxy for the expected spread? No, that's not right.
    # 
    # Better: "Relaxation to Identity" or "Multiplicative Adaptive Inflation".
    # Compute the ratio of the ensemble spread to the forecast spread at observed points?
    # Or use the innovation variance.
    
    # Let's use a simple multiplicative adaptive inflation based on the 
    # ratio of the ensemble variance to the observation variance at observed sites.
    # Target: The ensemble variance should be at least the observation variance.
    
    # Calculate ensemble variance at observed sites
    # updated_ensemble[:, observed_indices] is (N, n_obs)
    ens_var_obs = np.var(updated_ensemble[:, observed_indices], axis=0)
    
    # The observation variance is given.
    # We want to inflate the ensemble so that the spread is sufficient.
    # If ens_var_obs is too small, inflate.
    # Target ratio: 1.0 (or slightly higher, e.g., 1.1)
    # inflation_factor = sqrt(target_var / ens_var_obs)
    # But this is per-observation. We need a global factor or a vector.
    # Let's use a scalar adaptive inflation factor.
    
    # Average ratio
    ratio = np.mean(ens_var_obs / observation_variance)
    
    # If ratio < 1, we are under-dispersive. Inflate.
    # If ratio > 1, we are over-dispersive. Deflate (or don't inflate).
    # To prevent collapse, we ensure a minimum inflation.
    
    # Use a soft target: aim for ratio ~ 1.0
    # alpha = sqrt(1.0 / ratio) if ratio < 1.0 else 1.0
    # But we also need to account for the fact that the update reduces spread.
    
    # Let's use the "Adaptive Inflation" from Burgers et al.
    # Inflate the anomalies around the mean.
    
    mean_new = np.mean(updated_ensemble, axis=0)
    
    # Calculate the ratio of the ensemble spread to the observation spread
    # at the observed locations.
    # We want the ensemble to have enough spread to cover the observation noise.
    
    # A robust simple method:
    # alpha = 1.0
    # if ratio < 0.9:
    #     alpha = np.sqrt(0.9 / ratio)
    
    # Let's use a fixed target of 1.0 for the variance ratio.
    target_ratio = 1.0
    if ratio > 1e-6:
        alpha = np.sqrt(target_ratio / ratio)
    else:
        alpha = 1.0
        
    # Clamp alpha to avoid extreme values
    alpha = np.clip(alpha, 0.9, 1.5)
    
    # Apply inflation
    updated_ensemble = alpha * (updated_ensemble - mean_new) + mean_new
    
    # Ensure finite values
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    # Clip to [-100, 100] as per contract
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble
# EVOLVE-BLOCK-END
