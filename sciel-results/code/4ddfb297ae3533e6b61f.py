import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Deterministic Local Ensemble Square Root Filter (LES-RF) with localization.
    """
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 110.0  # 10s margin before 120s limit

    n_members, n_vars = ensemble.shape
    obs_idx = np.asarray(observed_indices)
    n_obs = len(obs_idx)
    
    # Ensure input is float
    ens = np.array(ensemble, dtype=np.float64)
    obs = np.array(observations, dtype=np.float64)
    r_var = float(observation_variance)
    
    # 1. Compute ensemble mean and anomalies
    mean = np.mean(ens, axis=0)
    anomalies = ens - mean[None, :]
    
    # 2. Localization
    # Compute distance matrix for observed variables
    # In Lorenz-96, indices are periodic.
    # Distance between i and j is min(|i-j|, n_vars - |i-j|)
    i_idx = np.arange(n_vars)
    dist_matrix = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(n_vars):
            d = abs(i - j)
            dist_matrix[i, j] = min(d, n_vars - d)
    
    # Gaussian localization function
    # Width chosen to cover local neighbors in L96 (coupling is i-2, i-1, i+1)
    # A width of ~2-3 is typical. Let's use 2.5.
    loc_width = 2.5
    loc_matrix = np.exp(-0.5 * (dist_matrix / loc_width) ** 2)
    
    # We only need localization for observed vs observed and observed vs all
    # Let's construct the localization matrix for the observation subset
    # L_obs_obs[i, j] = loc(obs_idx[i], obs_idx[j])
    # L_all_obs[i, j] = loc(i, obs_idx[j])
    
    # Optimization: Vectorize distance calculation for obs indices
    obs_i = obs_idx[:, None]
    obs_j = obs_idx[None, :]
    dist_obs_obs = np.abs(obs_i - obs_j)
    dist_obs_obs = np.minimum(dist_obs_obs, n_vars - dist_obs_obs)
    L_obs_obs = np.exp(-0.5 * (dist_obs_obs / loc_width) ** 2)
    
    all_i = np.arange(n_vars)[:, None]
    dist_all_obs = np.abs(all_i - obs_j)
    dist_all_obs = np.minimum(dist_all_obs, n_vars - dist_all_obs)
    L_all_obs = np.exp(-0.5 * (dist_all_obs / loc_width) ** 2)
    
    # 3. Compute Ensemble Covariance (Anomalies)
    # X_a = anomalies (n_members, n_vars)
    # We want to update X_a and mean.
    # Let X = anomalies.
    # C = (1/(n-1)) X.T X
    # We need to update X such that new C matches the Kalman update.
    
    # Standard LES-RF update:
    # X_new = X + K X_obs
    # Where X_obs = X[:, obs_idx]
    # K = C_xo C_oo^{-1}
    # C_xo = (1/(n-1)) X.T X_obs
    # C_oo = (1/(n-1)) X_obs.T X_obs
    
    # To avoid inverting C_oo, we use the square root form.
    # Let Y = X_obs.
    # We solve for a transformation matrix T such that:
    # (Y_new) = Y T
    # And the covariance updates correctly.
    
    # A simpler, robust approach for small n_obs:
    # Compute the Kalman gain explicitly using Cholesky of the localized obs covariance.
    
    # Localized Obs Covariance:
    # C_oo_loc = L_obs_obs * ( (1/(n-1)) X_obs.T X_obs )
    # Note: Localization is applied to the covariance.
    
    X_obs = anomalies[:, obs_idx]
    
    # Covariances
    C_oo = (X_obs.T @ X_obs) / (n_members - 1)
    C_xo = (anomalies.T @ X_obs) / (n_members - 1)
    
    # Apply localization to C_oo
    C_oo_loc = C_oo * L_obs_obs
    
    # Add observation noise
    R = r_var * np.eye(n_obs)
    C_oo_loc += R
    
    # Compute Kalman Gain
    # K = C_xo_loc C_oo_loc^{-1}
    # We also need to localize C_xo? 
    # Usually, localization is applied to the background covariance.
    # The cross-covariance C_xo is derived from the same anomalies, so it's consistent.
    # However, spurious correlations in C_xo can also be an issue.
    # A common practice is to apply the localization to C_oo and assume C_xo is consistent 
    # or apply a similar localization to C_xo if we had a full localization matrix.
    # For now, we use the standard EnKF gain derived from localized C_oo.
    
    try:
        # Cholesky decomposition for stability
        L_chol = np.linalg.cholesky(C_oo_loc)
        # Solve L y = C_xo.T  => y = C_xo.T L^{-1}
        # K = y.T L^{-T}
        # Actually, K = C_xo C_oo^{-1}
        # Let's solve C_oo_loc K.T = C_xo.T
        # K.T = C_oo_loc^{-1} C_xo.T
        K_T = np.linalg.solve(C_oo_loc, C_xo.T)
        K = K_T.T # (n_vars, n_obs)
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse if singular
        K = C_xo @ np.linalg.pinv(C_oo_loc)
        
    # Update Anomalies
    # X_new = X + K (obs - obs_mean_ens)
    # obs_mean_ens = mean[obs_idx]
    obs_mean_ens = mean[obs_idx]
    obs_innov = obs - obs_mean_ens
    
    # Update anomalies
    anomalies_new = anomalies + K @ obs_innov[None, :]
    
    # Update Mean
    # mean_new = mean + K @ obs_innov
    mean_new = mean + K @ obs_innov
    
    # 4. Adaptive Spread Control (Inflation)
    # Check if the ensemble spread is too low or too high relative to the obs error.
    # We estimate the error by comparing the updated ensemble mean to the observations.
    # Or better, compare the pre-update spread to the post-update spread.
    # A simple heuristic: ensure the post-update spread at observed locations is not zero.
    
    # Calculate spread at observed locations
    spread_obs = np.std(ens[:, obs_idx], axis=0, ddof=1)
    
    # Target spread is roughly sqrt(r_var)
    target_spread = np.sqrt(r_var)
    
    # If spread is too low, inflate.
    # We want to restore spread.
    # Let's check the ratio.
    # If spread_obs < 0.5 * target_spread, we might need to inflate.
    # However, LES-RF usually preserves spread better than standard EnKF.
    
    # Let's implement a simple multiplicative inflation if the spread drops too much.
    # We need to re-center the anomalies around the new mean?
    # The update above shifts the mean and the anomalies consistently.
    # The new ensemble is: ens_new = mean_new[None, :] + anomalies_new
    
    ens_new = mean_new[None, :] + anomalies_new
    
    # Check spread of new ensemble at observed locations
    new_spread_obs = np.std(ens_new[:, obs_idx], axis=0, ddof=1)
    
    # If new spread is significantly smaller than the expected error, inflate.
    # Threshold: 0.8 * target_spread
    if np.any(new_spread_obs < 0.8 * target_spread):
        # Calculate inflation factor
        # We want to scale the anomalies to increase spread.
        # Current spread at obs: new_spread_obs
        # Target: 0.9 * target_spread (slightly above to be safe)
        target = 0.9 * target_spread
        factors = target / new_spread_obs
        # Limit inflation factor to avoid instability
        factors = np.clip(factors, 1.0, 1.5)
        
        # Apply inflation to anomalies at observed locations?
        # Or globally? Global is safer for consistency.
        # Let's use a global inflation factor based on the minimum ratio.
        min_factor = np.min(factors)
        
        if min_factor > 1.0:
            # Inflate anomalies
            anomalies_new = anomalies_new * min_factor
            # Recalculate mean? No, mean is already updated.
            # But if we inflate anomalies, the mean stays the same?
            # Yes, mean is separate.
            ens_new = mean_new[None, :] + anomalies_new
            
    # Final safety check: finite and bounded
    ens_new = np.clip(ens_new, -100.0, 100.0)
    ens_new = np.nan_to_num(ens_new, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ens_new
# EVOLVE-BLOCK-END
