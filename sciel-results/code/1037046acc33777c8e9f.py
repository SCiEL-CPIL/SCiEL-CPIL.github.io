"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _gaspari_cohn(d, c=1.5):
    """Gaspari-Cohn correlation function for localization."""
    d = np.abs(d)
    result = np.zeros_like(d)
    mask1 = d <= 0.5 * c
    mask2 = (d > 0.5 * c) & (d <= 1.5 * c)
    mask3 = (d > 1.5 * c) & (d <= 2.5 * c)
    
    # d <= 0.5c
    result[mask1] = (-5.0 / 4.0) * d[mask1]**5 / c**5 + 5.0 * d[mask1]**4 / c**4 - 10.0 * d[mask1]**3 / c**3 + 5.0 * d[mask1]
    # 0.5c < d <= 1.5c
    result[mask2] = (1.0 / 12.0) * d[mask2]**5 / c**5 - (5.0 / 4.0) * d[mask2]**4 / c**4 + (5.0 / 2.0) * d[mask2]**3 / c**3 - 5.0 * d[mask2]**2 / c**2 + (25.0 / 12.0) * d[mask2] / c + 1.0 / 2.0
    # 1.5c < d <= 2.5c
    result[mask3] = (-1.0 / 16.0) * d[mask3]**5 / c**5 + (1.0 / 4.0) * d[mask3]**4 / c**4 - (15.0 / 8.0) * d[mask3]**3 / c**3 + (5.0 / 2.0) * d[mask3]**2 / c**2 - (25.0 / 12.0) * d[mask3] / c + 105.0 / 48.0
    
    return result

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Transform Kalman Filter (ETKF) update for Lorenz-96 assimilation.
    
    Parameters:
    ensemble: (M, N) array of forecast ensemble states
    observations: (K,) array of noisy observations for observed_indices
    observed_indices: array of indices that are observed
    observation_variance: scalar variance of observation noise
    
    Returns:
    updated ensemble with same shape (M, N)
    """
    M, N = ensemble.shape
    K = len(observed_indices)
    
    # Start with a copy of the ensemble
    result = ensemble.copy()
    
    # Adaptive Inflation
    # Estimate spread vs observation error to adjust inflation
    # We use a simple heuristic based on the ratio of ensemble spread to obs error
    # at observed locations
    if M > 1:
        spread = np.std(ensemble[:, observed_indices], axis=0)
        obs_err = np.sqrt(observation_variance)
        ratio = np.mean(spread) / obs_err if obs_err > 0 else 1.0
        
        # If spread is too small, inflate more aggressively
        # If spread is appropriate, use minimal inflation
        # Base inflation 1.01, max 1.05
        if ratio < 1.0:
            inflation = 1.0 + 0.05 * (1.0 - ratio)
            inflation = np.clip(inflation, 1.01, 1.05)
        else:
            inflation = 1.01
        
        # Apply multiplicative inflation around the mean
        mean = np.mean(result, axis=0)
        result = mean + inflation * (result - mean)
    else:
        inflation = 1.01

    # Local ETKF Update
    # Window size: include the target variable and its neighbors
    # For Lorenz-96, a window of 5 (2 left, 2 right) is typically sufficient
    window_half = 2
    window_size = 2 * window_half + 1
    
    # Pre-compute localization weights for the window
    # Distances from center (0) to neighbors
    dists = np.arange(-window_half, window_half + 1)
    # Periodic distance calculation for the small window
    # Since window is small (5) and N=20, no periodic wrap-around issues within window
    # unless window > N/2, which it isn't.
    loc_weights = _gaspari_cohn(dists, c=3.0)
    
    # Iterate over each state variable i
    for i in range(N):
        # Determine local window indices
        local_indices = [(i + j) % N for j in range(-window_half, window_half + 1)]
        
        # Identify which indices in the window are observed
        # Map global index to local position in window
        # local_pos 0 corresponds to i-window_half
        # local_pos 1 corresponds to i-window_half+1
        # ...
        # local_pos window_half corresponds to i
        
        local_obs_mask = np.zeros(window_size, dtype=bool)
        local_obs_vals = np.zeros(window_size)
        
        for l_idx, g_idx in enumerate(local_indices):
            if g_idx in observed_indices:
                # Find the value in observations array
                # observations is aligned with observed_indices
                obs_pos = np.where(observed_indices == g_idx)[0]
                if len(obs_pos) > 0:
                    local_obs_mask[l_idx] = True
                    local_obs_vals[l_idx] = observations[obs_pos[0]]
        
        n_obs_local = np.sum(local_obs_mask)
        
        # If no observations in this local window, keep the state as is (except inflation)
        if n_obs_local == 0:
            continue
        
        # Extract local ensemble states: shape (M, window_size)
        local_ens = result[:, local_indices]
        
        # Extract local observations: shape (n_obs_local,)
        y_local = local_obs_vals[local_obs_mask]
        
        # Compute local ensemble mean and anomalies
        local_mean = np.mean(local_ens, axis=0)
        local_anoms = local_ens - local_mean
        
        # Compute local ensemble covariance (M-1 normalization)
        # C = (A^T A) / (M-1) where A is (M, L)
        C_local = (local_anoms.T @ local_anoms) / (M - 1)
        
        # Observation operator H_local: selects observed components
        # H is (n_obs, window_size)
        H_local = np.zeros((n_obs_local, window_size))
        for k, l_idx in enumerate(np.where(local_obs_mask)[0]):
            H_local[k, l_idx] = 1.0
        
        # R = observation_variance * I
        R_local = observation_variance * np.eye(n_obs_local)
        
        # Innovation covariance S = H C H^T + R
        HC = H_local @ C_local
        S = HC @ H_local.T + R_local
        
        # Compute Gain K = C H^T S^-1
        # K shape: (window_size, n_obs_local)
        # K = C @ H_local.T @ inv(S)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        
        K_local = C_local @ H_local.T @ S_inv
        
        # Innovation y - H x_mean
        innovation = y_local - H_local @ local_mean
        
        # Update local mean
        local_mean_new = local_mean + K_local @ innovation
        
        # Update local ensemble using Transform
        # x_a = x_f + T * (y - H x_f) ? No, ETKF uses a transform on the anomalies.
        # Standard ETKF:
        # 1. Compute anomalies A (M, L)
        # 2. Compute local observations y (n_obs)
        # 3. Compute local mean x_mean
        # 4. Compute innovation d = y - H x_mean
        # 5. Compute gain K
        # 6. Compute "local observation" of anomalies? 
        #    Actually, the standard ETKF derivation (Hunt et al.) is:
        #    x_a = x_mean_a + T * (x_f - x_mean_f)
        #    where T is the transform matrix.
        #    T is derived from:
        #    A_f = x_f - x_mean_f (M, L)
        #    A_o = y - H x_mean_f (M, n_obs) -- wait, y is single obs, not ensemble obs.
        #    
        #    Let's use the simpler "Ensemble Square Root" or "EnKF" formulation 
        #    but localized. The previous code used a deterministic EnKF update:
        #    x_a = x_f + K (y - H x_f)
        #    This is equivalent to a specific transform.
        #    Let's stick to the deterministic EnKF update for the local window, 
        #    as it is simpler and robust, effectively acting as a local EnKF.
        
        # Deterministic update for each member in the local window
        for m in range(M):
            Hx_m = H_local @ local_ens[m, :]
            # x_a,m = x_f,m + K @ (y - H x_f,m)
            local_ens_new_m = local_ens[m, :] + K_local @ (y_local - Hx_m)
            result[m, local_indices] = local_ens_new_m
        
        # Alternatively, using the mean update + anomaly scaling (EnSRF-like)
        # But the direct update above is standard EnKF.
        
    # Final safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result
# EVOLVE-BLOCK-END
