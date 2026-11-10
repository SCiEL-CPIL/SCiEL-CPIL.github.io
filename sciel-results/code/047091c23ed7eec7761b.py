"""Local Ensemble Square Root Filter (ESRF) for Lorenz-96 assimilation."""
import numpy as np

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform a Local Ensemble Square Root Filter (ESRF) update.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors.
    observations : ndarray, shape (10,)
        Noisy observations of the observed indices.
    observed_indices : array-like
        Indices of the observed state components.
    observation_variance : float
        Variance of the observation noise.
        
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    n_ens = ensemble.shape[0]
    n_state = ensemble.shape[1]
    
    # Work with a copy
    result = ensemble.copy()
    
    # Convert observed_indices to a list for indexing
    obs_idx_list = list(observed_indices)
    
    # Sequential ESRF: process each observation one at a time
    for j, idx in enumerate(obs_idx_list):
        obs_val = observations[j]
        
        # Current ensemble mean for this component
        x_mean = result[:, idx].mean()
        
        # Current ensemble anomalies for this component
        x_anom = result[:, idx] - x_mean
        
        # Ensemble variance (using n_ens - 1 for unbiased estimate)
        x_var = np.sum(x_anom ** 2) / (n_ens - 1)
        
        # Innovation
        innovation = obs_val - x_mean
        
        # Observation variance
        r_var = observation_variance
        
        # Kalman gain
        k = x_var / (x_var + r_var)
        
        # Ensure k is in valid range
        k = np.clip(k, 0.0, 1.0)
        
        # New ensemble mean
        x_mean_new = x_mean + k * innovation
        
        # Apply ESRF update to each member
        # The ESRF update for a single observation:
        # x_i_new = x_mean_new + sqrt(1-k) * (x_i - x_mean) + (1-k) * innovation * sign(x_i - x_mean) / sqrt(1-k)
        
        for i in range(n_ens):
            a_i = x_anom[i]
            if abs(a_i) > 1e-12:
                sign_a = 1.0 if a_i > 0 else -1.0
                result[i, idx] = x_mean_new + a_i * np.sqrt(1 - k) + (1 - k) * innovation * sign_a / np.sqrt(1 - k)
            else:
                result[i, idx] = x_mean_new
    
    # Clip to valid range
    result = np.clip(result, -100.0, 100.0)
    
    return result
# EVOLVE-BLOCK-END
