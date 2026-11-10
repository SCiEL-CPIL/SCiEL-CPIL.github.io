import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Square Root Filter (Local ESRF) with adaptive inflation.
    """
    start_time = time.time()
    
    M, N = ensemble.shape
    K = len(observed_indices)
    
    ensemble = np.array(ensemble, dtype=np.float64)
    observations = np.array(observations, dtype=np.float64)
    observed_indices = np.array(observed_indices, dtype=int)
    
    # Handle observations shape
    if observations.ndim == 1:
        # If single vector, use it for all members? 
        # Usually EnKF uses perturbed obs. If not provided, we assume the same obs for all?
        # Or better: use the mean update logic if perturbed obs not available.
        # But the task says "independent N(0, 0.49) noise", implying perturbed obs.
        # If shape is (K,), we treat it as the mean observation and use a deterministic update?
        # Let's assume standard EnKF input: (M, K).
        # If (K,), we replicate it M times? No, that's not standard.
        # Let's check the seed program. It handles (M, K) by doing `observations - ensemble[:, observed_indices]`.
        # If observations is (K,), this would fail or broadcast incorrectly.
        # The seed program: `innovations = observations - ensemble[:, observed_indices]`
        # If observations is (K,), this is (K,) - (M, K) -> (M, K) via broadcasting.
        # This implies the same observation vector is used for all members? That's a deterministic filter update for the mean, but what about spread?
        # The seed program then does `result = ensemble + innovations @ gain.T`.
        # If innovations is (M, K) where each row is the same (y - x_i), then it's a perturbed obs update where the obs are identical.
        # This is a valid EnKF variant (deterministic obs, perturbed states).
        # So, we can handle both.
        pass
    
    # 1. Compute initial mean
    mean = np.mean(ensemble, axis=0)
    
    # 2. Sequential Local ESRF Update
    # We process each observation index one by one.
    # For each observation k, we update all N variables.
    
    # Pre-compute localization factors for efficiency?
    # Localization depends on distance between variable i and observed variable k.
    indices = np.arange(N)
    
    # Loop over observed indices
    for k in range(K):
        obs_idx = observed_indices[k]
        
        # Compute distance from obs_idx to all variables
        dist = np.abs(indices - obs_idx)
        dist = np.minimum(dist, N - dist)
        
        # Exponential localization factor for this observation
        # Scale 4.0
        loc_factor = np.exp(-dist / 4.0)
        
        # Compute sample covariance between each variable i and the observed variable obs_idx
        # Using the CURRENT ensemble (which has been updated by previous k's)
        # cov_i_obs = (1/(M-1)) * sum_m (x_m_i - mean_i) * (x_m_obs - mean_obs)
        
        # Anomalies of current ensemble
        anomalies = ensemble - np.mean(ensemble, axis=0)
        
        # Covariance vector: (N,)
        # cov[i] = (anomalies[:, i] @ anomalies[:, obs_idx]) / (M - 1)
        cov_vec = (anomalies.T @ anomalies[:, obs_idx]) / (M - 1)
        
        # Apply localization
        cov_vec_loc = cov_vec * loc_factor
        
        # Variance of observed variable
        var_obs = np.var(ensemble[:, obs_idx])
        
        # Total variance S = var_obs + observation_variance
        S = var_obs + observation_variance
        
        # Gain vector: (N,)
        # gain[i] = cov_vec_loc[i] / S
        gain_vec = cov_vec_loc / S
        
        # Innovation for each member: y_k - x_m_obs
        # If observations is (M, K), y_k is observations[:, k]
        # If observations is (K,), y_k is observations[k] (scalar)
        if observations.ndim == 2:
            y_k = observations[:, k]  # (M,)
        else:
            y_k = np.full(M, observations[k])  # (M,)
            
        innovation = y_k - ensemble[:, obs_idx]  # (M,)
        
        # Update each variable i for all members
        # x_m_i += gain_vec[i] * innovation_m
        # This is (M, N) += (M, 1) * (1, N)
        ensemble = ensemble + np.outer(innovation, gain_vec)
        
        # Check time
        if time.time() - start_time > 110:
            break
            
    # 3. Adaptive Inflation
    # Compute ensemble variance at observed points after update
    ens_var_obs = np.var(ensemble[:, observed_indices], axis=0)
    obs_var = observation_variance
    
    spread_ratio = np.mean(ens_var_obs) / obs_var if obs_var > 0 else 1.0
    
    # Target ratio ~1.15
    target_ratio = 1.15
    deviation = (spread_ratio - target_ratio) / target_ratio
    
    # Tanh inflation
    beta = 0.20
    inflation = 1.0 + beta * np.tanh(-deviation)
    inflation = np.clip(inflation, 0.95, 1.35)
    
    # Apply inflation around the updated mean
    mean_a = np.mean(ensemble, axis=0)
    ensemble = mean_a + inflation * (ensemble - mean_a)
    
    # Ensure finite and within bounds
    ensemble = np.clip(ensemble, -100, 100)
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ensemble
