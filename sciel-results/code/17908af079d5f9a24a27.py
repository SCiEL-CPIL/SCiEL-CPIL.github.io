"""Local Ensemble Kalman Filter with Adaptive Inflation and Cosine Localization for Lorenz-96."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Kalman Filter (EnKF) with adaptive inflation and cosine localization.
    
    This update handles the non-Gaussian nature of the Lorenz-96 attractor better than 
    a global Gaussian update by processing observations sequentially in local windows.
    It uses a spatially adaptive inflation scheme to prevent filter divergence due to 
    under-dispersion, which is common with small ensembles.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of even-indexed states
    - observed_indices: array of observed indices (even indices)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble with same (16, 20) shape
    """
    t0 = time.time()
    
    nm, nvars = ensemble.shape  # 16, 20
    nobs = len(observed_indices)  # 10
    obs_var = float(observation_variance)
    
    # Ensure finite input
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    ensemble = np.clip(ensemble, -100.0, 100.0)
    
    updated_ensemble = ensemble.copy()
    
    # Pre-compute local indices and weights for each observation
    # We use a local window of radius 5 (covering 11 variables) for 20 total variables.
    # This is wide enough to capture correlations but narrow enough to avoid spurious long-range links.
    local_radius = 5
    
    # Cosine taper function for localization
    def cosine_taper(dist, radius):
        if dist >= radius:
            return 0.0
        return 0.5 * (1 + np.cos(np.pi * dist / radius))
    
    # Process each observation sequentially
    for k, obs_idx in enumerate(observed_indices):
        obs_val = observations[k]
        obs_idx_int = int(obs_idx)
        
        # Determine local window indices
        local_indices = []
        local_weights = []
        for d in range(-local_radius, local_radius + 1):
            idx = (obs_idx_int + d) % nvars
            dist = abs(d)
            weight = cosine_taper(dist, local_radius)
            if weight > 1e-6:
                # Avoid duplicates in the list if window wraps around (not an issue for radius < 10 in n=20)
                if idx not in local_indices:
                    local_indices.append(idx)
                    local_weights.append(weight)
        
        local_indices = np.array(local_indices)
        local_weights = np.array(local_weights)
        n_local = len(local_indices)
        
        if n_local == 0:
            continue
            
        # Extract local ensemble
        local_ens = updated_ensemble[:, local_indices]
        local_mean = np.mean(local_ens, axis=0)
        local_anoms = local_ens - local_mean
        
        # Compute local covariance C = (1/(nm-1)) * A^T A
        # Apply localization taper to the covariance matrix
        # C_tapered[i,j] = C[i,j] * w_i * w_j ? Or just use the windowing.
        # Standard EnKF localization often just restricts the update to the window.
        # However, we can also taper the anomalies themselves or the gain.
        
        # Find the position of the observed variable in the local window
        try:
            center_pos = np.where(local_indices == obs_idx_int)[0][0]
        except IndexError:
            continue
            
        # Compute Innovation
        innovation = obs_val - local_mean[center_pos]
        
        # Compute Ensemble Variance at the observed location
        ens_var_obs = np.var(local_ens[:, center_pos], ddof=1)
        
        # --- Adaptive Inflation ---
        # Estimate if we need to inflate.
        # If the innovation is large relative to the expected innovation variance, 
        # the ensemble is likely under-dispersive.
        # Expected innovation variance = ens_var_obs + obs_var
        # We want to scale the anomalies such that the new spread matches the innovation better.
        
        # A common adaptive inflation strategy:
        # alpha = max(1.0, 1.0 + 0.1 * (innovation^2 / (ens_var_obs + obs_var) - 1))
        # But this can be unstable.
        
        # Let's use a spatially adaptive inflation factor.
        # If ens_var_obs is very small compared to obs_var + innovation^2, inflate more.
        # Target: ens_var_new ~ ens_var_obs * alpha
        # We want the innovation to be "explained" by the spread.
        # Let's use a simple heuristic:
        # If |innovation| > 2 * sqrt(ens_var_obs + obs_var), we are likely under-dispersive.
        
        inflation_factor = 1.0
        if ens_var_obs > 1e-6:
            z_score = abs(innovation) / np.sqrt(ens_var_obs + obs_var)
            if z_score > 1.5:
                # Scale inflation with the excess z-score
                inflation_factor = 1.0 + 0.1 * (z_score - 1.5)
                # Cap inflation to prevent instability
                inflation_factor = np.clip(inflation_factor, 1.0, 2.0)
        else:
            # If spread is zero, we must inflate significantly to avoid division by zero later
            inflation_factor = 1.5
            
        # Apply inflation to local anomalies
        # We only inflate the anomalies in the local window
        local_anoms_inflated = local_anoms * np.sqrt(inflation_factor)
        
        # Re-compute local mean (should be same) and anomalies
        local_ens_inflated = local_mean + local_anoms_inflated
        
        # Compute Kalman Gain
        # P_oo = C_oo + R
        # C_oo is the variance of the observed variable in the inflated ensemble
        C_oo = np.var(local_ens_inflated[:, center_pos], ddof=1)
        P_oo = C_oo + obs_var
        
        if P_oo < 1e-12:
            P_oo = 1e-12
            
        # K_i = C_ii / P_oo
        # C_ii is the covariance between variable i and the observed variable
        C_col = np.cov(local_ens_inflated, rowvar=False, ddof=1)[:, center_pos]
        
        # Apply localization taper to the gain?
        # The gain K_i determines how much variable i is updated.
        # We can multiply K_i by the localization weight w_i to soften the update.
        K_local = (C_col / P_oo) * local_weights
        
        # Update local ensemble members
        for m in range(nm):
            H_x_f = local_ens_inflated[m, center_pos]
            innov_m = obs_val - H_x_f
            # x_a = x_f + K * innov_m
            # Note: We are updating the inflated forecast to get the analysis.
            # local_ens[m] is the original forecast.
            # We should update the original forecast using the gain derived from the inflated forecast?
            # Standard EnKF: Inflate, Compute Gain, Update Original Forecast.
            # Or: Inflate, Update Inflated Forecast, Deflate? No, usually update the inflated state.
            # Let's update the inflated state directly.
            update_vec = K_local * innov_m
            local_ens_inflated[m] = local_ens_inflated[m] + update_vec
            
        # Write back to global ensemble
        # We use the updated inflated ensemble as the new analysis
        updated_ensemble[:, local_indices] = local_ens_inflated
        
        # Deadline check
        if time.time() - t0 > 115.0:
            break
            
    # --- Final Safety Checks ---
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated_ensemble
# EVOLVE-BLOCK-END
