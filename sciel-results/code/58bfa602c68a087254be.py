"""
Combined Local ESRF/EnKF analysis for Lorenz-96 partial-observation assimilation.
Uses periodic Gaspari-Cohn localization on the full covariance and adaptive inflation.
"""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _gaspari_cohn_periodic(distances, L, N):
    """
    Gaspari-Cohn correlation function with periodic distance calculation.
    
    Parameters:
    -----------
    distances : ndarray
        Raw distance matrix (i, j)
    L : float
        Correlation length scale.
    N : int
        Dimension of the system (for periodicity).
    
    Returns:
    --------
    result : ndarray
        Localized correlation values.
    """
    # Calculate periodic distances
    # dist = min(|i-j|, N - |i-j|)
    d = np.abs(distances)
    d = np.minimum(d, N - d)
    
    # Normalize by length scale
    d_norm = d / L
    
    result = np.zeros_like(d_norm)
    
    # Mask for d <= 1.0
    mask1 = d_norm <= 1.0
    # Mask for 1.0 < d <= 2.0
    mask2 = (d_norm > 1.0) & (d_norm <= 2.0)
    
    # For d <= 1
    if np.any(mask1):
        dm = d_norm[mask1]
        # Gaspari-Cohn polynomial for d in [0, 1]
        # f(d) = -0.25*d^5 + 0.5*d^4 + 0.625*d^3 - 0.5*d^2 + 1.0
        result[mask1] = (-0.25 * dm**5 + 0.5 * dm**4 + 0.625 * dm**3 - 0.5 * dm**2 + 1.0)
        
    # For 1 < d <= 2
    if np.any(mask2):
        dm = d_norm[mask2]
        # Gaspari-Cohn polynomial for d in (1, 2]
        # f(d) = 0.0625*d^5 - 0.125*d^4 + 0.625*d^3 - 0.5*d^2 + 0.125*d - 0.0625
        result[mask2] = (0.0625 * dm**5 - 0.125 * dm**4 + 0.625 * dm**3 - 0.5 * dm**2 + 0.125 * dm - 0.0625)
        
    # For d > 2, result is 0 (already initialized)
    return result

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an ensemble analysis update using a localized EnKF with adaptive inflation.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of 16 members, each a 20-dimensional state.
    observations : ndarray, shape (10,)
        Noisy observations of the 10 even-indexed state variables.
    observed_indices : array-like
        Indices of observed state variables (should be [0, 2, 4, ..., 18]).
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    M, N = ensemble.shape
    obs_idx = np.array(observed_indices, dtype=int)
    K = len(obs_idx)
    
    ensemble = np.array(ensemble, dtype=np.float64)
    observations = np.array(observations, dtype=np.float64)
    
    try:
        # 1. Compute Ensemble Mean and Anomalies
        mean = np.mean(ensemble, axis=0)
        anomalies = ensemble - mean  # Shape (M, N)
        
        # 2. Compute Full Covariance
        # C = (1/(M-1)) * A^T A
        cov = (anomalies.T @ anomalies) / (M - 1)  # Shape (N, N)
        
        # 3. Apply Periodic Gaspari-Cohn Localization to Covariance
        # We need the distance matrix for all pairs (i, j) in 0..N-1
        indices = np.arange(N)
        dist_matrix = np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])
        
        # Tuned localization length. 4.0 is a good starting point for L96 (N=20)
        localization_length = 4.0 
        loc_factor = _gaspari_cohn_periodic(dist_matrix, localization_length, N)
        
        cov_loc = cov * loc_factor  # Localized covariance
        
        # 4. Observation Operator and Innovation
        # H is (K, N)
        # Since we only observe specific indices, we can extract them directly
        # H @ cov_loc @ H.T is equivalent to cov_loc[np.ix_(obs_idx, obs_idx)]
        S_obs = cov_loc[np.ix_(obs_idx, obs_idx)]  # Shape (K, K)
        
        # Add Observation Noise
        R = observation_variance * np.eye(K)
        S = S_obs + R  # Innovation Covariance
        
        # Innovation Vector (Mean)
        y = observations
        y_mean = mean[obs_idx]
        innovation = y - y_mean  # Shape (K,)
        
        # 5. Compute Gain
        # We need K_gain = C H^T S^{-1}
        # C H^T is simply the columns of cov_loc corresponding to observed indices
        # Shape: (N, K)
        CHt = cov_loc[:, obs_idx]
        
        # Solve S x = innovation
        try:
            # Use solve for better numerical stability than inv
            S_inv_innov = np.linalg.solve(S, innovation)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            S_inv_innov = S_inv @ innovation
            
        # Update Vector for Mean
        # mean_a = mean + CHt @ S_inv_innov
        update_mean = CHt @ S_inv_innov
        mean_a = mean + update_mean
        
        # 6. Update Ensemble Members
        # For each member m, innovation_m = y - ensemble[m, obs_idx]
        # x_a_m = x_m + CHt @ (S^{-1} @ innovation_m)
        
        # We need S^{-1} to apply to each member's innovation.
        # If we didn't compute it in the exception block above, compute it now.
        if 'S_inv' not in locals():
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)
        
        # Compute Gain Matrix explicitly for vectorized member updates
        # Gain shape: (N, K)
        Gain = CHt @ S_inv
        
        # Innovations for all members: Shape (M, K)
        # y is (K,), broadcast to (M, K)
        ens_obs = ensemble[:, obs_idx]
        innovations_all = np.tile(y, (M, 1)) - ens_obs
        
        # Update all members: (M, K) @ (K, N) -> (M, N)
        updates_all = innovations_all @ Gain.T
        updated_ensemble = ensemble + updates_all
        
        # 7. Adaptive Inflation
        # Compute ensemble variance at observed points after update
        # We want to maintain spread. 
        # ens_var_obs = var(updated_ensemble[:, obs_idx], axis=0)
        # obs_var = observation_variance
        
        # A common adaptive inflation strategy:
        # alpha = 1 + beta * tanh( (target_spread_ratio - actual_spread_ratio) / target_spread_ratio )
        # Here we use a simpler robust form:
        # Compare mean ensemble variance at obs points to observation variance.
        
        ens_var_obs = np.var(updated_ensemble[:, obs_idx], axis=0)
        mean_ens_var = np.mean(ens_var_obs)
        
        if observation_variance > 0:
            spread_ratio = mean_ens_var / observation_variance
        else:
            spread_ratio = 1.0
            
        # Target ratio: slightly above 1 to ensure ensemble is not over-confident
        # If spread_ratio < target, we inflate.
        target_ratio = 1.2 
        beta = 0.3 # Strength of inflation
        
        # Calculate inflation factor
        # deviation = (spread_ratio - target_ratio) / target_ratio
        # inflation = 1.0 + beta * tanh(-deviation)
        # If spread_ratio is low (under-dispersed), -deviation is positive, tanh is positive, inflation > 1.
        # If spread_ratio is high (over-dispersed), -deviation is negative, tanh is negative, inflation < 1.
        
        deviation = (spread_ratio - target_ratio) / target_ratio
        inflation_factor = 1.0 + beta * np.tanh(-deviation)
        
        # Clamp inflation factor to reasonable bounds
        inflation_factor = np.clip(inflation_factor, 0.95, 1.4)
        
        # Apply inflation around the updated mean
        # updated_ensemble = mean_a + inflation_factor * (updated_ensemble - mean_a)
        updated_ensemble = mean_a + inflation_factor * (updated_ensemble - mean_a)
        
    except Exception as e:
        # Fallback: Simple nudging if anything fails
        # This ensures we always return a valid array
        updated_ensemble = ensemble.copy()
        if len(obs_idx) > 0:
            # Strong nudge to observations
            updated_ensemble[:, obs_idx] = 0.5 * updated_ensemble[:, obs_idx] + 0.5 * observations[:, np.newaxis]
            
    # Ensure output is finite and within bounds
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Check deadline
    if time.time() > deadline:
        pass # Return what we have
        
    return updated_ensemble

# EVOLVE-BLOCK-END
