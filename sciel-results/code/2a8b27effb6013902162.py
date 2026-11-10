import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) update with exponential localization and tanh-based adaptive inflation.
    
    This implementation combines the structural benefits of smooth exponential localization
    (which handles periodic boundaries and long-range correlations more gracefully than 
    polynomial cutoffs for small ensembles) with a robust tanh-based inflation scheme 
    to prevent filter divergence.
    
    Parameters:
    ensemble: (M, N) array of ensemble states, M=16, N=20
    observations: (M, K) array where K is number of observed indices
    observed_indices: array of indices that are observed
    observation_variance: scalar variance of observation noise
    
    Returns:
    Updated ensemble with same (M, N) shape
    """
    start_time = time.time()
    
    M, N = ensemble.shape
    K = len(observed_indices)
    
    # Ensure inputs are numpy arrays of float64
    ensemble = np.array(ensemble, dtype=np.float64)
    observations = np.array(observations, dtype=np.float64)
    observed_indices = np.array(observed_indices, dtype=int)
    
    # Compute ensemble mean
    mean = np.mean(ensemble, axis=0)
    
    # Compute anomaly matrix (M, N)
    anomalies = ensemble - mean
    
    # Covariance: (N, N)
    # Using M-1 for unbiased estimate
    cov = (anomalies.T @ anomalies) / (M - 1)
    
    # Localization: Smooth exponential with periodic distance
    # Scale 4.0 was found to be optimal in high-scoring variants
    localization_scale = 4.0
    indices = np.arange(N)
    dist = np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])
    # Periodic distance for Lorenz-96 ring
    dist = np.minimum(dist, N - dist)
    
    # Exponential localization factor
    loc_factor = np.exp(-dist / localization_scale)
    
    # Localize the covariance
    cov_loc = cov * loc_factor
    
    # Observation operator: H is (K, N)
    # Since we are observing specific indices, H is a selection matrix
    H = np.zeros((K, N))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # Observation covariance R
    R = observation_variance * np.eye(K)
    
    # Handle observations shape
    # If observations is 1D, it's a single observation vector (unlikely for EnKF with M>1, 
    # but possible if perturbed obs are not provided). 
    # The task contract says "observations of half its coordinates" and "independent N(0,0.49) noise".
    # Usually, this implies perturbed observations for each ensemble member, i.e., shape (M, K).
    # If shape is (K,), we treat it as a single truth-like observation and use standard EnKF mean update.
    # If shape is (M, K), we use perturbed observations.
    
    if observations.ndim == 1:
        y_vec = observations
        y_mean = y_vec
        obs_matrix = np.tile(y_vec, (M, 1))
    else:
        y_mean = np.mean(observations, axis=0)
        obs_matrix = observations
    
    # Innovation vector for mean update: (K,)
    innovation_mean = y_mean - H @ mean
    
    # Innovation covariance: S = H C H^T + R
    S = H @ cov_loc @ H.T + R
    
    # Compute Gain K_gain = C H^T S^-1
    # Instead of inverting S, we solve S X = innovation to get the update vector directly.
    # This is more numerically stable.
    try:
        # We need gain = cov_loc @ H.T @ S_inv
        # update_vector = gain @ innovation_mean = cov_loc @ H.T @ S_inv @ innovation_mean
        # Let Z = S_inv @ innovation_mean. Solve S Z = innovation_mean.
        Z = np.linalg.solve(S, innovation_mean)
        update_vector = cov_loc @ H.T @ Z
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if S is singular
        S_inv = np.linalg.pinv(S)
        Z = S_inv @ innovation_mean
        update_vector = cov_loc @ H.T @ Z
    
    # Update mean
    mean_a = mean + update_vector
    
    # Update ensemble members
    # For perturbed observations (M, K):
    # x_a_i = x_f_i + gain @ (y_i - H x_f_i)
    # We need the full gain matrix (N, K) for this.
    # gain = cov_loc @ H.T @ S_inv
    if 'S_inv' not in locals():
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
    
    gain = cov_loc @ H.T @ S_inv  # (N, K)
    
    # Innovations for each member: (M, K)
    innovations = obs_matrix - ensemble[:, observed_indices]
    
    # Update each member: x_a = x_f + gain @ innovation
    # result[i] = ensemble[i] + gain @ innovations[i]
    # In matrix form: result = ensemble + innovations @ gain.T
    result = ensemble + innovations @ gain.T
    
    # Adaptive Inflation:
    # We want to maintain the ensemble spread relative to the observation error.
    # Compute ensemble variance at observed points
    ens_var_obs = np.var(ensemble[:, observed_indices], axis=0)  # (K,)
    obs_var = observation_variance
    
    # Compute average spread ratio
    if obs_var > 0:
        spread_ratio = np.mean(ens_var_obs) / obs_var
    else:
        spread_ratio = 1.0
    
    # Tanh-based adaptive inflation
    # Target ratio ~1.15 (slightly over-dispersed to account for model error/non-Gaussianity)
    target_ratio = 1.15
    deviation = (spread_ratio - target_ratio) / target_ratio
    
    # Beta controls the strength of the inflation correction
    beta = 0.20
    inflation = 1.0 + beta * np.tanh(-deviation)
    
    # Clamp inflation to reasonable bounds to prevent explosion or collapse
    inflation = np.clip(inflation, 0.95, 1.35)
    
    # Apply inflation around the updated mean
    # This ensures the mean remains consistent while adjusting the spread
    result = mean_a + inflation * (result - mean_a)
    
    # Ensure the result is finite and within bounds
    result = np.clip(result, -100, 100)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 110:
        return result
    
    return result
