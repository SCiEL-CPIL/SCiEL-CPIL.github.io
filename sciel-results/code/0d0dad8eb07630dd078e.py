import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) with Gaspari-Cohn localization and Spatially Adaptive Inflation.
    
    Parameters:
    ensemble: (M, N) array of ensemble states, M=16, N=20
    observations: (M, K) array of perturbed observations
    observed_indices: (K,) array of indices observed
    observation_variance: float, variance of observation noise
    
    Returns:
    updated_ensemble: (M, N) array of updated ensemble states
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    ensemble = np.array(ensemble, dtype=np.float64)
    observations = np.array(observations, dtype=np.float64)
    observed_indices = np.array(observed_indices, dtype=int)
    
    M, N = ensemble.shape
    K = len(observed_indices)
    
    # 1. Compute Ensemble Mean and Anomalies
    mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean
    
    # 2. Compute Covariance
    # C = (1/(M-1)) * A^T A
    cov = (anomalies.T @ anomalies) / (M - 1)
    
    # 3. Localization (Gaspari-Cohn)
    # Periodic distance on a ring of size N
    indices = np.arange(N)
    dist = np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])
    dist = np.minimum(dist, N - dist)
    
    # Gaspari-Cohn function parameters
    # r is distance, L is half-width (scale)
    L = 3.0  # Localization scale, typical for L96 is 3-5
    r = dist
    
    # Gaspari-Cohn function
    # f(r) = 1 - 0.5 * (5*r^5 - 5*r^4 + 2*r^3) / L^3  for 0 <= r <= L
    # f(r) = 2 - 0.5 * (5*(r-L)^5 / (2L)^3 + 5*(r-L)^4 / (2L)^2 + 10*(r-L)^3 / (2L) + 4*(r-L)^2 / (2L) + 0.125*(r-L))  for L < r <= 2L
    # f(r) = 0 for r > 2L
    
    # Vectorized implementation
    loc_factor = np.zeros_like(r)
    mask1 = r <= L
    mask2 = (r > L) & (r <= 2*L)
    
    # For 0 <= r <= L
    r1 = r[mask1]
    loc_factor[mask1] = 1 - 0.5 * (5*(r1/L)**5 - 5*(r1/L)**4 + 2*(r1/L)**3)
    
    # For L < r <= 2L
    r2 = r[mask2]
    r2_norm = (r2 - L) / (2*L)
    loc_factor[mask2] = 2 - 0.5 * (5*r2_norm**5 + 5*r2_norm**4 + 10*r2_norm**3 + 4*r2_norm**2 + 0.125*r2_norm)
    
    # Localize Covariance
    cov_loc = cov * loc_factor
    
    # 4. Observation Operator H
    H = np.zeros((K, N))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # 5. Observation Covariance R
    R = observation_variance * np.eye(K)
    
    # 6. Innovation and Gain
    # Use mean of perturbed observations for the mean update
    y_mean = np.mean(observations, axis=0)
    innovation = y_mean - H @ mean
    
    # Innovation Covariance S = H C_loc H^T + R
    S = H @ cov_loc @ H.T + R
    
    # Calculate Gain K_gain = C_loc H^T S^-1
    try:
        # Solve S X = H^T to get S^-1 H^T, then multiply by C_loc
        # Or just invert S if K is small (K=10)
        S_inv = np.linalg.inv(S)
        gain = cov_loc @ H.T @ S_inv
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        gain = cov_loc @ H.T @ S_inv
    
    # 7. Update Ensemble
    # Standard EnKF with perturbed observations
    # x_a_i = x_f_i + gain @ (y_i - H x_f_i)
    innovations = observations - ensemble[:, observed_indices]  # (M, K)
    
    # result = ensemble + innovations @ gain.T
    result = ensemble + innovations @ gain.T
    
    # 8. Spatially Adaptive Inflation (SAI)
    # Inflate the ensemble at observed points if the spread is too low.
    # Compute ensemble variance at observed points after update
    ens_var_obs = np.var(result[:, observed_indices], axis=0)
    
    # Target variance is roughly the observation variance, or slightly less.
    # If ens_var < target, inflate.
    target_var = observation_variance * 0.8  # Slightly conservative target
    
    # Inflation factor per observed index
    # alpha_i = sqrt(target_var / ens_var_i) if ens_var_i < target_var else 1.0
    # To avoid over-inflation, cap the factor.
    alpha = np.ones(K)
    valid = ens_var_obs > 1e-6
    alpha[valid] = np.sqrt(target_var[valid] / ens_var_obs[valid])
    
    # Cap inflation to prevent instability
    alpha = np.clip(alpha, 1.0, 1.5)
    
    # Apply inflation to the ensemble at observed indices
    # We inflate the anomalies at observed indices
    result_mean = np.mean(result, axis=0)
    result_anoms = result - result_mean
    
    # Inflate anomalies at observed indices
    for j, idx in enumerate(observed_indices):
        result_anoms[:, idx] *= alpha[j]
        
    result = result_mean + result_anoms
    
    # 9. Final Safety Checks
    result = np.clip(result, -100, 100)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 110:
        return result
        
    return result
