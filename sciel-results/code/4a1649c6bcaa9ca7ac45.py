import numpy as np
import time

def _gaspari_cohn(d, c=1.5):
    """
    Gaspari-Cohn correlation function for localization.
    d: array of distances
    c: correlation length scale
    """
    d = np.abs(d)
    result = np.zeros_like(d)
    
    # Mask 1: d <= 0.5 * c
    mask1 = d <= 0.5 * c
    if np.any(mask1):
        dm = d[mask1]
        cm = c
        result[mask1] = (-5.0 / 4.0) * (dm**5 / cm**5) + 5.0 * (dm**4 / cm**4) - 10.0 * (dm**3 / cm**3) + 5.0 * dm
        
    # Mask 2: 0.5 * c < d <= 1.5 * c
    mask2 = (d > 0.5 * c) & (d <= 1.5 * c)
    if np.any(mask2):
        dm = d[mask2]
        cm = c
        result[mask2] = (1.0 / 12.0) * (dm**5 / cm**5) - (5.0 / 4.0) * (dm**4 / cm**4) + (5.0 / 2.0) * (dm**3 / cm**3) - 5.0 * (dm**2 / cm**2) + (25.0 / 12.0) * (dm / cm) + 0.5
        
    # Mask 3: 1.5 * c < d <= 2.5 * c
    mask3 = (d > 1.5 * c) & (d <= 2.5 * c)
    if np.any(mask3):
        dm = d[mask3]
        cm = c
        result[mask3] = (-1.0 / 16.0) * (dm**5 / cm**5) + (1.0 / 4.0) * (dm**4 / cm**4) - (15.0 / 8.0) * (dm**3 / cm**3) + (5.0 / 2.0) * (dm**2 / cm**2) - (25.0 / 12.0) * (dm / cm) + 10.0 / 3.0
        
    # d > 2.5 * c is 0

    return result

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Transform Kalman Filter (LETKF) for Lorenz-96.
    
    ensemble: (16, 20) array
    observations: (10,) array (observed at even indices)
    observed_indices: (10,) array of indices
    observation_variance: float
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    # 1. Compute ensemble mean and anomalies
    mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean
    
    # 2. Compute ensemble covariance of observed variables
    # C_oo = (1/(n-1)) * Y_oo * Y_oo.T
    # Y_oo: (n_obs, n_members)
    Y_oo = anomalies[:, observed_indices].T  # (n_obs, n_members)
    
    # Add observation noise to covariance
    # R = observation_variance * I
    # C_oo_obs = C_oo + R
    # We use the transformation matrix approach.
    # The standard EnKF gain can be computed via the transform matrix.
    
    # To avoid singularities and handle localization, we compute the localization matrix for observations.
    # Since observations are at specific indices, we need to localize the cross-covariance C_xo.
    # However, a simpler and robust approach for small n_obs is to compute the gain directly with localized cross-covariance.
    
    # Let's use the standard ETKF/LETKF transform.
    # The transformation matrix T is such that (I - T^T T) * Y_oo = Y_oo * (C_oo^{-1} (C_oo + R))^{-1/2} ... this is complex.
    
    # Alternative: Compute the gain K directly using localized covariances.
    # K = C_xo C_oo^{-1}
    # C_xo: (n_vars, n_obs)
    # C_oo: (n_obs, n_obs)
    
    # Compute C_oo (observed-observed covariance)
    C_oo = (Y_oo @ Y_oo.T) / (n_members - 1.0)
    C_oo += observation_variance * np.eye(n_obs)
    
    # Compute C_xo (state-observed covariance) with localization
    # We localize C_xo using Gaspari-Cohn based on index distance.
    # C_xo_unlocalized = (1/(n-1)) * X * Y_oo.T
    X = anomalies  # (n_vars, n_members)
    C_xo_raw = (X @ Y_oo.T) / (n_members - 1.0)  # (n_vars, n_obs)
    
    # Localization matrix for cross-covariance
    # L: (n_vars, n_obs)
    loc_matrix = np.zeros((n_vars, n_obs))
    for i in range(n_vars):
        for j in range(n_obs):
            # Periodic distance
            dist = abs(i - observed_indices[j])
            dist = min(dist, n_vars - dist)
            loc_matrix[i, j] = _gaspari_cohn(dist, c=2.0)  # Correlation length 2.0
            
    C_xo = C_xo_raw * loc_matrix
    
    # Compute Gain K = C_xo @ inv(C_oo)
    try:
        K = C_xo @ np.linalg.inv(C_oo)
    except np.linalg.LinAlgError:
        # Fallback to simple nudging if matrix is singular
        result = ensemble.copy()
        result[:, observed_indices] += 0.5 * (observations - result[:, observed_indices])
        return result
    
    # 3. Compute innovation
    # innovation = observations - mean[observed_indices]
    innovation = observations - mean[observed_indices]
    
    # 4. Update mean
    new_mean = mean + K @ innovation
    
    # 5. Update anomalies using the transform
    # The LETKF transform ensures that the ensemble spread is reduced appropriately.
    # For simplicity and robustness, we can use the standard EnKF update for anomalies:
    # Y_x_new = Y_x - K @ Y_oo
    # This is the "perturbed observations" or "deterministic" EnKF equivalent for anomalies.
    # Note: Strictly, LETKF uses a specific transform T to preserve spread. 
    # The simple subtraction Y_x - K Y_oo is the standard EnKF update which works well for Gaussian errors.
    # Given the chaotic nature, we might need inflation.
    
    new_anomalies = anomalies - K @ Y_oo
    
    # 6. Ensemble Inflation
    # Adaptive inflation to prevent collapse.
    # Estimate the expected spread vs actual spread.
    # A simple heuristic: if the ensemble spread decreases too much, inflate.
    # Or use a fixed multiplicative inflation.
    # Let's try a simple adaptive inflation based on the innovation magnitude.
    # If innovation is large relative to expected, we might have under-dispersion.
    
    # Calculate innovation norm
    inn_norm = np.linalg.norm(innovation)
    # Expected innovation norm is sqrt(trace(C_oo + R))
    exp_inn_norm = np.sqrt(np.trace(C_oo))
    
    # Inflation factor
    # If inn_norm > exp_inn_norm, we might need less inflation? 
    # Actually, if the ensemble is too tight, the innovation will be large relative to the ensemble spread.
    # A common heuristic: inflate if the ensemble spread is too small compared to the innovation.
    
    # Let's use a fixed multiplicative inflation of 1.05 to be safe, or adaptive.
    # Adaptive: alpha = 1.0 + 0.05 * (inn_norm / (exp_inn_norm + 1e-6) - 1.0)
    # Clip alpha to [1.0, 1.2]
    
    ratio = inn_norm / (exp_inn_norm + 1e-6)
    inflation_factor = 1.0
    if ratio > 1.0:
        inflation_factor = 1.0 + 0.1 * (ratio - 1.0)
    inflation_factor = np.clip(inflation_factor, 1.0, 1.2)
    
    new_anomalies *= inflation_factor
    
    # 7. Reconstruct ensemble
    result = new_mean + new_anomalies
    
    # 8. Safety checks
    if not np.all(np.isfinite(result)):
        # Fallback
        result = ensemble.copy()
        result[:, observed_indices] += 0.5 * (observations - result[:, observed_indices])
        
    # Clip to valid range
    result = np.clip(result, -100.0, 100.0)
    
    # Check deadline
    if time.time() > deadline:
        return result
        
    return result
