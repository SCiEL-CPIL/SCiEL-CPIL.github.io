"""Ensemble Kalman Filter for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman filter analysis update for Lorenz-96.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of even-indexed states
    - observed_indices: array of observed indices (even indices)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble with same (16, 20) shape
    """
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin for 120s limit
    
    n_members, n_state = ensemble.shape
    n_obs = len(observed_indices)
    
    # 1. Compute ensemble mean and anomalies
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean
    
    # 2. Compute ensemble covariance
    # Covariance of the full state (n_state x n_state)
    # Using (N-1) in denominator for unbiased estimate
    C = (anomalities.T @ anomalies) / (n_members - 1)
    
    # 3. Localization
    # Apply a gasparre-cohn localization function to the covariance
    # This is critical for small ensembles to avoid spurious correlations
    loc_width = 4.0  # Correlation length scale in grid points
    indices = np.arange(n_state)
    # Distance matrix for periodic domain
    dist = np.abs(indices[:, None] - indices[None, :])
    dist = np.minimum(dist, n_state - dist)  # Periodic distance
    
    # Gasparre-Cohn localization function
    # r is the distance
    r = dist / loc_width
    loc_func = np.zeros_like(r)
    mask1 = r < 1
    mask2 = (r >= 1) & (r < 2)
    
    loc_func[mask1] = (1.0 / 15.0) * (
        -4 * r[mask1]**5 + 5 * r[mask1]**4 + 5 * r[mask1]**2 - 1
    )
    loc_func[mask2] = (1.0 / 15.0) * (
        -4 * r[mask2]**5 + 25 * r[mask2]**4 - 60 * r[mask2]**3 + 40 * r[mask2]**2 + 5
    )
    
    # Apply localization to covariance
    C_loc = C * loc_func
    
    # 4. Compute Kalman Gain
    # Observation operator H selects observed indices
    # H C H^T is the covariance of observations
    H_C_H_t = C_loc[np.ix_(observed_indices, observed_indices)]
    
    # Innovation covariance: S = H C H^T + R
    R = observation_variance
    S = H_C_H_t + R * np.eye(n_obs)
    
    # Kalman Gain: K = C H^T S^-1
    # We need C H^T, which is the covariance between full state and observations
    C_H_t = C_loc[:, observed_indices]
    
    # Solve for K
    try:
        S_inv = np.linalg.inv(S)
        K = C_H_t @ S_inv
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular
        S_inv = np.linalg.pinv(S)
        K = C_H_t @ S_inv

    # 5. Innovation
    # y - H x_mean
    obs_mean = np.mean(observations)  # Observations are already centered? No, they are raw.
    # Wait, observations are (10,) array of noisy observations.
    # The ensemble mean at observed indices is ens_mean[observed_indices]
    innovation = observations - ens_mean[observed_indices]
    
    # 6. Adaptive Inflation
    # Estimate inflation factor to maintain consistency
    # If the ensemble spread is too small compared to the innovation, inflate
    # Simple adaptive inflation: 
    # Let sigma_obs = sqrt(diag(S))
    # Let sigma_innov = std(innovation)
    # If sigma_innov > sigma_obs, we might need to inflate
    # A common heuristic: inflation factor alpha such that 
    # (1/2) * (innovation^T S^-1 innovation) ~ chi2(n_obs)
    # For now, use a fixed small inflation or a simple adaptive one.
    # Let's use a simple adaptive inflation based on the ratio of observed spread to expected spread.
    
    # Expected innovation variance is trace(S)/n_obs
    # Actual innovation variance is var(innovation)
    expected_innov_var = np.trace(S) / n_obs
    actual_innov_var = np.var(innovation)
    
    # If actual > expected, the ensemble is too tight. Inflate.
    # If actual < expected, the ensemble is too spread. Deflate (or keep 1).
    # Use a simple multiplicative inflation.
    # alpha = sqrt(actual / expected)
    # But we need to be careful not to explode.
    # Let's use a smoothed version.
    
    # Standard adaptive inflation (Anderson et al.)
    # We want to find alpha such that the ensemble covariance after inflation matches the innovation.
    # A simpler approach often used:
    # alpha = 1 + 0.5 * (actual_innov_var / expected_innov_var - 1)
    # Clamp alpha to [1.0, 1.5]
    
    if expected_innov_var > 0:
        ratio = actual_innov_var / expected_innov_var
        # Smooth the ratio to avoid jumps
        # If ratio is 1, alpha=1. If ratio is 2, alpha=1.5?
        # Let's use alpha = sqrt(ratio) clipped.
        alpha = np.sqrt(ratio)
        alpha = np.clip(alpha, 1.0, 1.5)
    else:
        alpha = 1.0

    # Apply inflation to the ensemble mean and anomalies
    # Inflate around the ensemble mean
    ens_mean_inflated = ens_mean
    anomalies_inflated = anomalies * alpha
    
    # Recalculate mean after inflation? 
    # Usually, inflation is applied to the anomalies around the mean.
    # The mean itself is not changed by multiplicative inflation around the mean.
    # However, the covariance changes.
    
    # Update the ensemble
    # x_new = x_old + K * (y - H x_mean)
    # But we inflated the anomalies.
    # The update formula for each member i:
    # x_i_new = x_mean + (x_i - x_mean) * alpha + K * (y - x_mean)
    # Wait, the Kalman gain K was computed with the original covariance C_loc.
    # If we inflate, the covariance changes, so K should ideally change.
    # For simplicity and stability, we often apply the gain computed from the pre-inflation covariance 
    # to the innovation, and then apply inflation to the resulting ensemble spread.
    # Or, more commonly in practice:
    # 1. Inflate the ensemble before computing statistics.
    # Let's restart the logic with inflation first.

    # --- Revised Logic with Inflation First ---
    
    # 1. Inflate Ensemble
    # Use a fixed inflation for stability or adaptive.
    # Let's stick to the adaptive calculation above but apply it to the anomalies.
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean
    
    # Apply inflation to anomalies
    anomalies_inflated = anomalies * alpha
    ensemble_inflated = ens_mean + anomalies_inflated
    
    # 2. Recompute Statistics with Inflated Ensemble
    C_infl = (anomalies_inflated.T @ anomalies_inflated) / (n_members - 1)
    
    # 3. Localization
    C_loc_infl = C_infl * loc_func
    
    # 4. Kalman Gain
    H_C_H_t = C_loc_infl[np.ix_(observed_indices, observed_indices)]
    S = H_C_H_t + R * np.eye(n_obs)
    C_H_t = C_loc_infl[:, observed_indices]
    
    try:
        S_inv = np.linalg.inv(S)
        K = C_H_t @ S_inv
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        K = C_H_t @ S_inv

    # 5. Innovation
    innovation = observations - ens_mean[observed_indices]
    
    # 6. Update Ensemble
    # x_new = x_inflated + K * innovation
    # K is (n_state, n_obs), innovation is (n_obs,)
    # So K * innovation is (n_state,)
    # We add this to each member.
    update = K @ innovation
    
    result = ensemble_inflated + update[np.newaxis, :]
    
    # 7. Post-processing: Ensure finite and within bounds
    # Replace NaN/Inf with 0 or mean
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
        
    # Clip to [-100, 100]
    result = np.clip(result, -100.0, 100.0)
    
    return result

# EVOLVE-BLOCK-END
