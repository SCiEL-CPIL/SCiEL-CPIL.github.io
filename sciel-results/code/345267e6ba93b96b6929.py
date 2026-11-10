import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) update with smooth exponential localization and adaptive inflation.
    """
    start_time = time.time()
    
    M, N = ensemble.shape
    K = len(observed_indices)
    
    ensemble = np.array(ensemble, dtype=np.float64)
    observations = np.array(observations, dtype=np.float64)
    observed_indices = np.array(observed_indices, dtype=int)
    
    # Compute ensemble mean
    mean = np.mean(ensemble, axis=0)
    
    # Compute anomaly matrix (M, N)
    anomalies = ensemble - mean
    
    # Covariance: (N, N)
    cov = (anomalies.T @ anomalies) / (M - 1)
    
    # Localization: smooth exponential with periodic distance
    # Use a scale that preserves meaningful correlations in Lorenz-96
    localization_scale = 4.5
    indices = np.arange(N)
    dist = np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])
    dist = np.minimum(dist, N - dist)
    
    # Exponential localization - smooth, no hard cutoff
    loc_factor = np.exp(-dist / localization_scale)
    
    # Localize the covariance
    cov_loc = cov * loc_factor
    
    # Observation operator: H is (K, N)
    H = np.zeros((K, N))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # Observation covariance R
    R = observation_variance * np.eye(K)
    
    # Innovation: y - H x
    y_mean = np.mean(observations, axis=0)
    
    # Innovation vector: (K,)
    innovation = y_mean - H @ mean
    
    # Innovation covariance: S = H C H^T + R
    S = H @ cov_loc @ H.T + R
    
    # Gain: K_gain = C H^T S^-1
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    gain = cov_loc @ H.T @ S_inv
    
    # Update: x_a = x_f + K (y - H x_f)
    mean_a = mean + gain @ innovation
    
    # For the ensemble, use perturbed observations
    innovations = observations - ensemble[:, observed_indices]  # (M, K)
    
    # Update each member
    result = ensemble + innovations @ gain.T
    
    # Adaptive Inflation:
    # Compute ensemble variance at observed points
    ens_var_obs = np.var(ensemble[:, observed_indices], axis=0)  # (K,)
    obs_var = observation_variance
    
    # Compute spread ratio
    spread_ratio = np.mean(ens_var_obs) / obs_var if obs_var > 0 else 1.0
    
    # Smooth adaptive inflation using a target ratio of ~1.2
    target_ratio = 1.2
    # Use a smooth function to determine inflation
    # If spread_ratio < target, inflate; if > target, mild deflation
    # Use tanh-like smooth transition
    deviation = (spread_ratio - target_ratio) / target_ratio
    # Inflation factor: 1 + beta * tanh(-deviation) for smooth behavior
    beta = 0.15
    inflation = 1.0 + beta * np.tanh(-deviation)
    
    # Clamp to reasonable bounds
    inflation = np.clip(inflation, 0.95, 1.25)
    
    # Apply inflation around the updated mean
    result = mean_a + inflation * (result - mean_a)
    
    # Ensure the result is finite and within bounds
    result = np.clip(result, -100, 100)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 110:
        return result
    
    return result
