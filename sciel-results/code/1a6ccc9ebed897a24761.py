import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter update with Gaspari-Cohn localization and adaptive inflation.
    """
    start_time = time.time()
    
    M, N = ensemble.shape
    K = len(observed_indices)
    
    ensemble = np.array(ensemble, dtype=np.float64)
    observations = np.array(observations, dtype=np.float64)
    observed_indices = np.array(observed_indices, dtype=int)
    
    # Compute ensemble mean
    mean = np.mean(ensemble, axis=0)
    
    # Anomaly matrix
    anomalies = ensemble - mean  # (M, N)
    
    # Covariance: (N, N)
    cov = (anomalies.T @ anomalies) / (M - 1)
    
    # Gaspari-Cohn localization
    def gaspari_cohn(r, L):
        """Gaspari-Cohn correlation function with correlation length L."""
        r = np.abs(r)
        result = np.zeros_like(r)
        mask1 = r <= L / 2.0
        mask2 = (r > L / 2.0) & (r <= L)
        r1 = r[mask1]
        r2 = r[mask2]
        result[mask1] = (-5.0/4.0 * r1**5 / L**5 + 5.0/2.0 * r1**4 / L**4 
                         - 30.0/7.0 * r1**3 / L**3 + 5.0/3.0 * r1**2 / L**2 
                         + 4.0 - 2.0/3.0 * L / L)
        # Simplified: standard GC
        result[mask1] = (-5.0/4.0*(r1/L)**5 + 5.0/2.0*(r1/L)**4 
                        - 30.0/7.0*(r1/L)**3 + 5.0/3.0*(r1/L)**2 + 4.0 - 2.0/3.0)
        result[mask2] = (1.0/12.0*(r2/L)**5 - 5.0/4.0*(r2/L)**4 
                        + 5.0/3.0*(r2/L)**3 - 5.0/12.0*(r2/L) + 5.0/3.0)
        return result
    
    localization_length = 5.0  # grid points
    indices = np.arange(N)
    dist = np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])
    dist = np.minimum(dist, N - dist)  # periodic
    gc = gaspari_cohn(dist, localization_length)
    
    # Localize covariance
    cov_loc = cov * gc
    
    # Observation operator
    H = np.zeros((K, N))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # Observation covariance
    R = observation_variance * np.eye(K)
    
    # Use perturbed observation EnKF approach
    # Innovation for each member: y_i - H x_i
    innovations = observations - ensemble[:, observed_indices]  # (M, K)
    
    # Compute gain using the "gain form" with localized covariance
    # S = H C H^T + R
    HC = H @ cov_loc  # (K, N)
    S = HC @ H.T + R  # (K, K)
    
    # Compute S^{-1} via Cholesky for stability
    try:
        L_chol = np.linalg.cholesky(S)
        # Solve S x = b for each column of H^T
        # K_gain = C H^T S^{-1}
        # S^{-1} = L^{-T} L^{-1}
        # Solve L z = H^T -> z = L^{-1} H^T
        Ht = H.T  # (N, K)
        z = np.linalg.solve(L_chol, Ht)  # (N, K), z = L^{-1} H^T
        # Then S^{-1} H^T = L^{-T} z
        S_inv_Ht = np.linalg.solve(L_chol.T, z)  # (N, K)
        gain = cov_loc @ S_inv_Ht  # (N, K)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        try:
            S_inv = np.linalg.inv(S)
            gain = cov_loc @ H.T @ S_inv
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            gain = cov_loc @ H.T @ S_inv
    
    # Update each member: x_a_i = x_f_i + gain @ innovation_i
    # gain: (N, K), innovations: (M, K)
    result = ensemble + innovations @ gain.T
    
    # Adaptive inflation based on innovation statistics
    # Compute the normalized innovation squared
    mean_innovation = np.mean(innovations, axis=0)  # (K,)
    # Expected innovation variance: diag(S)
    expected_innov_var = np.diag(S)
    # Normalized innovation squared
    norm_innov_sq = np.sum((mean_innovation / np.sqrt(expected_innov_var + 1e-12))**2, axis=0)
    # Degrees of freedom
    dof = K
    # Inflation factor: sqrt(dof / norm_innov_sq) clipped
    # If norm_innov_sq > dof, ensemble is under-dispersed -> inflate
    if norm_innov_sq > 0:
        inflation = np.sqrt(dof / norm_innov_sq)
        # Clip to reasonable range
        inflation = np.clip(inflation, 1.0, 1.5)
    else:
        inflation = 1.0
    
    # Apply inflation around the analysis mean
    result_mean = np.mean(result, axis=0)
    result = result_mean + inflation * (result - result_mean)
    
    # Safety: ensure finite and bounded
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100, 100)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 115:
        return result
    
    return result
