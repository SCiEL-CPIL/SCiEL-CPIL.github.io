import numpy as np
import time
# EVOLVE-BLOCK-START

# Time deadline guard
_DEADLINE = 140.0
_t0 = None

def _check_deadline():
    """Return True if we should stop to avoid timeout."""
    global _t0
    if _t0 is None:
        _t0 = time.time()
    return (time.time() - _t0) > _DEADLINE

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an Ensemble Square Root Filter (EnSRF) analysis update.
    Uses sequential coordinate-wise regression for the update.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of observed values (even indices)
    - observed_indices: array of observed index positions
    - observation_variance: scalar observation error variance
    
    Returns:
    - Updated ensemble with same shape (16, 20)
    """
    n_members, n_state = ensemble.shape
    n_obs = len(observations)
    
    # Work with a copy
    ens = ensemble.astype(np.float64).copy()
    
    # Precompute statistics
    ens_mean = np.mean(ens, axis=0)
    
    # Observation operator indices
    obs_idx = observed_indices
    
    # We will update the ensemble sequentially for each observation.
    # For each observation j:
    # 1. Compute mean and std of the current ensemble member at obs_idx[j].
    # 2. Compute the regression of other states onto this state.
    # 3. Update the other states based on the innovation.
    # 4. Set the observed state to the observation (or adjust it).
    
    # To maintain consistency and avoid order dependence, we can use a "square root" 
    # or "regression" scheme. Here we use the standard sequential regression:
    # x_a = x_f + C(x_f, y_f) / Var(y_f) * (y_obs - y_f)
    # where C is covariance and Var is variance.
    # For an ensemble, this is equivalent to:
    # x_a_m = x_f_m + ( (x_f_m - x_f_mean) . (y_f_m - y_f_mean) / sum((y_f_m - y_f_mean)^2) ) * (y_obs - y_f_m)
    
    # However, we need to account for observation error.
    # The effective variance of the innovation is Var(y_f) + R.
    # The gain for the regression is Cov(x, y) / (Var(y) + R).
    
    # Localization
    decay_length = 1.8
    
    for j in range(n_obs):
        idx_j = obs_idx[j]
        
        # Current values for this observation index
        y_f = ens[:, idx_j]
        y_mean = np.mean(y_f)
        y_anom = y_f - y_mean
        
        # Variance of the forecast observation
        var_y = np.mean(y_anom ** 2) # Using N-1 or N? Usually N-1 for sample variance.
        # Let's use N-1 to be consistent with covariance estimation
        var_y = np.sum(y_anom ** 2) / (n_members - 1)
        
        # Effective variance including observation error
        var_y_eff = var_y + observation_variance
        
        if var_y_eff < 1e-10:
            continue
            
        # Innovation
        innovation = observations[j] - y_mean
        
        # For each other state i, compute the regression update
        # We update all states, including the observed one, but the observed one 
        # should converge to the observation if we treat it correctly.
        # Actually, in EnSRF, we often update the unobserved states and then 
        # replace the observed state with the observation (or a weighted average).
        # A common approach:
        # 1. Update unobserved states via regression.
        # 2. Replace observed state with observation.
        
        # Let's update all states i != idx_j first, then handle idx_j.
        # Or, update all states including idx_j with the regression formula, 
        # which naturally pulls idx_j towards the observation.
        
        for i in range(n_state):
            if i == idx_j:
                continue # Handle separately or include? 
                # If we include it:
                # Cov(y, y) / Var(y_eff) * (y_obs - y_f)
                # = Var(y) / (Var(y) + R) * (y_obs - y_f)
                # This is a partial update. 
                # Standard EnSRF often sets the observed variable to the observation 
                # or uses the full update. Let's use the full update for consistency 
                # and then maybe enforce the observation? 
                # Actually, the "regression" update for the observed variable itself 
                # results in:
                # y_a_m = y_f_m + (Var(y)/(Var(y)+R)) * (y_obs - y_f_m)
                # This is not exactly y_obs. 
                # To get the exact EnSRF result (equivalent to Kalman filter on mean/cov),
                # we should update the mean and covariance correctly.
                
                # Let's stick to the standard sequential regression for unobserved 
                # and set the observed to the observation? No, that increases spread 
                # artificially if not careful.
                
                # Better approach: 
                # Update all states i using:
                # x_i_a_m = x_i_f_m + K_i * (y_obs - y_f_m)
                # where K_i = Cov(x_i, y_j) / (Var(y_j) + R)
                
                # For i = idx_j:
                # K_j = Var(y_j) / (Var(y_j) + R)
                # y_j_a_m = y_j_f_m + K_j * (y_obs - y_j_f_m)
                # This preserves the mean and reduces variance.
                
                pass # We will do it in the loop below for all i, including j.
            
            x_f = ens[:, i]
            x_anom = x_f - np.mean(x_f)
            
            # Covariance between x_i and y_j
            cov_xy = np.sum(x_anom * y_anom) / (n_members - 1)
            
            # Localization
            d = abs(i - idx_j)
            d = min(d, n_state - d)
            w = np.exp(-d ** 2 / (2.0 * decay_length ** 2))
            
            # Gain
            k_i = w * cov_xy / var_y_eff
            
            # Update
            ens[:, i] = x_f + k_i * (observations[j] - y_f)
            
    # After sequential updates, the ensemble mean has been updated.
    # However, the sequential regression can sometimes lead to underestimation of 
    # uncertainty or other issues. Adaptive inflation is crucial.
    
    # Adaptive Inflation
    ens_mean_new = np.mean(ens, axis=0)
    anomalies_new = ens - ens_mean_new
    cov_new = (anomalies_new.T @ anomalies_new) / (n_members - 1)
    
    # Compute expected innovation variance for each observation
    # We need H P H^T + R
    # Since we updated sequentially, the current covariance is cov_new.
    # But the innovation was computed against the *forecast* mean.
    # For inflation, we compare the spread of the ensemble at observed locations 
    # to the observation error.
    
    # Simple spread matching:
    # For each observed index, compare ensemble variance to observation variance?
    # No, compare to the expected innovation variance if it were a forecast.
    # This is tricky after update.
    
    # Alternative: Global inflation based on the ratio of observed innovation 
    # magnitude to expected magnitude?
    
    # Let's use a simple global inflation factor based on the total spread 
    # relative to the observation noise.
    
    # Calculate the average variance of the ensemble at observed indices
    obs_vars = np.array([np.var(ens[:, idx], ddof=1) for idx in observed_indices])
    avg_obs_var = np.mean(obs_vars)
    
    # The "expected" variance if the filter were perfect is related to R.
    # But we want the ensemble to have enough spread.
    # If avg_obs_var < R, we might need to inflate.
    # If avg_obs_var > R, we might be over-dispersed.
    
    # A robust metric: 
    # The innovation norm should be consistent with S.
    # But we don't have the forecast ensemble anymore.
    
    # Let's use a simpler adaptive inflation:
    # If the ensemble spread at observed points is too small compared to R, inflate.
    # Target: avg_obs_var should be at least R (or some fraction).
    # Actually, the analysis variance should be < R.
    
    # Let's try to match the ensemble spread to the observation error variance 
    # if the spread is too low.
    
    # Inflation factor
    # If avg_obs_var is very small, inflate.
    # target_var = 0.5 * observation_variance # Example
    # This is heuristic.
    
    # Let's use the ratio of the innovation to the expected innovation variance 
    # from the *previous* step? No, we don't store it.
    
    # Let's use a fixed inflation with a small adaptive component.
    # Or, use the ratio of the sum of squared innovations to the trace of S.
    
    # Since we don't have the forecast ensemble, we can't compute S accurately 
    # from the current ensemble.
    
    # Let's use a simple global inflation based on the spread at observed locations.
    # If the spread is less than R, inflate to R.
    # If the spread is greater, maybe deflate?
    
    # Inflation = sqrt(R / avg_obs_var) ?
    # This forces the analysis spread to be R, which is wrong (analysis spread < R).
    
    # Let's use a constant inflation of 1.01-1.05 for stability, 
    # and adapt if the spread collapses.
    
    inflation = 1.01
    
    # Check for collapse
    if avg_obs_var < 0.1 * observation_variance:
        inflation = np.sqrt(observation_variance / (avg_obs_var + 1e-6))
        inflation = np.clip(inflation, 1.0, 2.0)
        
    ens = ens_mean_new + inflation * (ens - ens_mean_new)
    
    # Clip to valid range
    ens = np.clip(ens, -100.0, 100.0)
    
    # Ensure finiteness
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ens

# EVOLVE-BLOCK-END
