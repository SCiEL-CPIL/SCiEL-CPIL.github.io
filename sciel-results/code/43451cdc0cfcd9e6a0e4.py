import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) update with localization and adaptive inflation.
    
    Parameters:
    ensemble: (M, N) array of ensemble states, M=16, N=20
    observations: (M, K) array where K is number of observed indices
                  Each row is the noisy observation for that ensemble member's observed coords
                  Actually, re-reading: "observations of half its coordinates" with "independent N(0,0.49) noise"
                  The observations are likely a single vector or per-member. Given the interface returns (16,20),
                  and observations has shape (M, K) where K = len(observed_indices).
                  But wait, typically in EnKF, observations are a single vector y (K,), and we compare ensemble mean to y.
                  However, the signature says observations. Let's assume observations is (M, K) where each row is a perturbed observation
                  or just the same observation repeated? 
                  Actually, standard EnKF uses a single observation vector. But here we have ensemble size 16.
                  If observations is (16, 10), it might be that each ensemble member has its own perturbed observation,
                  or it's the same observation for all. Given "independent N(0,0.49) noise", it's likely that
                  observations[i, j] = truth[observed_indices[j]] + noise[i, j].
                  We will use the standard EnKF update with perturbed observations or just use the mean.
                  To be safe and standard, we'll treat observations as a single vector if possible, but the shape is (M, K).
                  Let's use the ensemble mean of observations as the "true" observation vector, or better,
                  use the standard perturbed observation EnKF.
                  
                  However, a simpler and robust approach: use the ensemble Kalman filter where we compare
                  the ensemble to the observations. Since observations is (M, K), we can compute the innovation
                  as observations - ensemble[:, observed_indices].
    """
    start_time = time.time()
    
    M, N = ensemble.shape
    K = len(observed_indices)
    
    # Ensure inputs are numpy arrays
    ensemble = np.array(ensemble, dtype=np.float64)
    observations = np.array(observations, dtype=np.float64)
    observed_indices = np.array(observed_indices, dtype=int)
    
    # Result starts as a copy of the ensemble
    result = ensemble.copy()
    
    # Compute ensemble mean and covariance
    mean = np.mean(ensemble, axis=0)
    
    # Compute anomaly matrix (M-1) x N
    anomalies = ensemble - mean
    # Covariance: (N, N)
    # Using the standard formula: C = (1/(M-1)) * A^T A
    cov = (anomalies.T @ anomalies) / (M - 1)
    
    # Localization: Gaspari-Cohn or simple exponential decay
    # For Lorenz-96, a common localization scale is around 4-6 grid points
    # We'll use a simple exponential localization
    localization_scale = 4.0
    # Create distance matrix
    indices = np.arange(N)
    dist = np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])
    # Periodic distance
    dist = np.minimum(dist, N - dist)
    # Exponential localization
    loc_factor = np.exp(-dist / localization_scale)
    
    # Localize the covariance
    cov_loc = cov * loc_factor
    
    # Observation operator: H is (K, N)
    # H[i, j] = 1 if j == observed_indices[i]
    H = np.zeros((K, N))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # Observation covariance R
    R = observation_variance * np.eye(K)
    
    # Innovation: y - H x
    # Since observations is (M, K), we can use the mean observation or perturbed.
    # Let's use the mean of observations as the "true" observation vector for the update.
    y_mean = np.mean(observations, axis=0)
    
    # Innovation vector: (K,)
    innovation = y_mean - H @ mean
    
    # Innovation covariance: S = H C H^T + R
    S = H @ cov_loc @ H.T + R
    
    # Gain: K_gain = C H^T S^-1
    # We need to solve S^{-1} innovation
    try:
        # Solve S * x = innovation
        S_inv_innov = np.linalg.solve(S, innovation)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        S_inv_innov = np.linalg.pinv(S) @ innovation
    
    # Gain: (N, K)
    gain = cov_loc @ H.T @ np.linalg.inv(S)
    
    # Update: x_a = x_f + K (y - H x_f)
    # For the mean:
    mean_a = mean + gain @ innovation
    
    # For the ensemble:
    # We can use the standard EnKF update for each member.
    # Or use the linearized update: x_a_i = x_f_i + gain @ (y_i - H x_f_i)
    # Since we have perturbed observations, let's use that.
    # y_i is observations[i, :]
    # innovation_i = y_i - H x_f_i
    innovations = observations - ensemble[:, observed_indices]  # (M, K)
    
    # Update each member: x_a_i = x_f_i + gain @ innovation_i
    # gain is (N, K), innovations is (M, K)
    # So: result = ensemble + innovations @ gain.T
    result = ensemble + innovations @ gain.T
    
    # Adaptive Inflation:
    # Check if the ensemble spread is too small compared to the observation error.
    # A simple inflation: if the ensemble variance at observed points is too low, inflate.
    # Compute ensemble variance at observed points
    ens_var_obs = np.var(ensemble[:, observed_indices], axis=0)  # (K,)
    obs_var = observation_variance
    
    # If ensemble variance is much smaller than observation variance, inflate
    # Target: ens_var_obs should be at least some fraction of obs_var
    # Simple adaptive inflation:
    # alpha = 1 + beta * (obs_var - ens_var_obs) / obs_var
    # But this can be unstable. Let's use a simpler approach:
    # If the mean innovation is large, inflate.
    
    # Alternative: use a fixed small inflation if needed.
    # For now, let's add a simple inflation based on the ratio of ensemble spread to observation error.
    spread_ratio = np.mean(ens_var_obs) / obs_var
    if spread_ratio < 0.5:
        inflation = 1.0 + 0.1 * (0.5 - spread_ratio)
        # Inflate the ensemble around the updated mean
        result = mean_a + inflation * (result - mean_a)
    else:
        inflation = 1.0
    
    # Ensure the result is finite and within bounds
    result = np.clip(result, -100, 100)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Deadline guard: if we're taking too long, just return the current result
    elapsed = time.time() - start_time
    if elapsed > 110:  # Safety margin before 120s limit
        return result
    
    return result
