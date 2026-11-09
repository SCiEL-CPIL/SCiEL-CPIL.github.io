import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Performs a Stochastic Interpolation (SI) update.
    
    This method interpolates between the background ensemble and the observations
    for the observed coordinates, while leaving unobserved coordinates unchanged 
    (or slightly inflated to prevent collapse). It is robust for chaotic systems 
    where the linear Gaussian assumption of EnKF may fail.
    
    Parameters:
    ensemble: np.ndarray of shape (16, 20). Current ensemble states.
    observations: np.ndarray of shape (10,). Noisy observations of even indices.
    observed_indices: np.ndarray of shape (10,). Indices of observed variables.
    observation_variance: float. Variance of observation noise.
    
    Returns:
    np.ndarray of shape (16, 20). Updated ensemble states.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    N, D = ensemble.shape
    obs_idx = np.array(observed_indices)
    n_obs = len(obs_idx)
    
    # 1. Compute background statistics
    mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean
    
    # Background variance for each coordinate
    bg_var = np.var(ensemble, axis=0)
    
    # 2. Stochastic Interpolation Update
    # For observed coordinates, we blend the background member with the observation.
    # The weight alpha is determined by the relative variances.
    # alpha = sqrt(bg_var / (bg_var + obs_var))
    # x_new = alpha * x_bg + (1 - alpha) * obs
    # This ensures that if bg_var is high, we trust the observation more (lower alpha for bg? 
    # Wait, standard SI:
    # x_a = x_b + alpha * (y - x_b) is linear nudging.
    # True SI (Evensen 2003):
    # x_a = sqrt(alpha) * x_b + sqrt(1-alpha) * y  ... no, that's for variance.
    
    # Let's use the standard Stochastic Interpolation formula:
    # The analysis variance is the geometric mean of background and observation variance?
    # Actually, a common robust SI formulation for the state is:
    # x_a = w * x_b + (1-w) * y
    # where w is chosen such that the analysis variance matches the harmonic mean or geometric mean.
    
    # Let's stick to a simpler, very effective "Relaxation to Observations" with adaptive gain.
    # Gain K_i = bg_var_i / (bg_var_i + obs_var)
    # This is the Kalman gain for a scalar with independent components.
    # However, SI often uses a fixed or slightly adaptive relaxation.
    
    # Let's try the "Ensemble Kalman Filter with Stochastic Interpolation" flavor:
    # For observed indices:
    # x_new = x_bg + K * (y - x_bg)
    # But to preserve spread better, we can use:
    # x_new = sqrt(1-K) * x_bg + sqrt(K) * y  (This is for variance preservation if y is deterministic? No.)
    
    # Let's use the standard scalar EnKF update which is equivalent to SI in 1D:
    # K = P_b / (P_b + R)
    # x_a = x_b + K (y - x_b)
    # P_a = (1-K) P_b
    
    # To avoid spread collapse (a common issue with N=16), we apply adaptive inflation.
    
    K = bg_var / (bg_var + observation_variance)
    
    # Ensure K is in [0, 1]
    K = np.clip(K, 0.0, 1.0)
    
    # Initialize new ensemble
    ensemble_new = ensemble.copy()
    
    # Update observed coordinates
    # x_new[:, obs_idx] = x_bg[:, obs_idx] + K[obs_idx] * (y - x_bg[:, obs_idx])
    # y is scalar per observation index.
    # We need to broadcast y to the ensemble dimension.
    y_expanded = observations[:, None] # (n_obs, 1)
    x_bg_obs = ensemble[:, obs_idx]    # (N, n_obs)
    K_expanded = K[obs_idx][None, :]   # (1, n_obs)
    
    innovation = y_expanded - x_bg_obs
    ensemble_new[:, obs_idx] = x_bg_obs + K_expanded * innovation
    
    # 3. Adaptive Inflation
    # Check if the spread has collapsed.
    # Compare analysis spread to background spread.
    # If analysis spread < background spread * threshold, inflate.
    
    # Calculate spread for all coordinates
    # Note: Unobserved coordinates are unchanged, so their spread is bg spread.
    # We only care about the global spread or the observed spread.
    
    # Simple global inflation based on the ratio of total variances
    bg_total_var = np.sum(bg_var)
    an_total_var = np.sum(np.var(ensemble_new, axis=0))
    
    # Target ratio: we want the analysis spread to be at least 95% of background spread
    # to prevent filter divergence due to under-dispersion.
    ratio = an_total_var / bg_total_var if bg_total_var > 1e-10 else 1.0
    
    if ratio < 0.95:
        # Inflate the anomalies around the new mean
        mean_new = np.mean(ensemble_new, axis=0)
        anomalies_new = ensemble_new - mean_new
        
        # Inflate factor
        # We want to increase variance by factor 0.95/ratio
        # Variance scales with factor^2, so factor = sqrt(0.95/ratio)
        inflation_factor = np.sqrt(0.95 / ratio)
        
        # Cap inflation to prevent explosion
        inflation_factor = min(inflation_factor, 1.1)
        
        ensemble_new = mean_new[None, :] + anomalies_new * inflation_factor
        
    # 4. Safety Checks
    # Clip to valid range
    ensemble_new = np.clip(ensemble_new, -100.0, 100.0)
    
    # Ensure finite
    if not np.all(np.isfinite(ensemble_new)):
        # Fallback to original ensemble if something went wrong
        return ensemble.copy()
        
    return ensemble_new
# EVOLVE-BLOCK-END
