import numpy as np
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Iterative Ensemble Kalman Filter (Non-linear EnKF) for Lorenz-96.
    Iterates the linear update to better handle non-linearities.
    """
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=int)
    
    n_particles, n_state = ensemble.shape
    n_obs = len(observed_indices)
    
    # Observation operator indices
    H_idx = observed_indices
    
    # Observation noise covariance (scalar variance for all obs)
    R_var = observation_variance
    
    # Iterative EnKF (Non-linear EnKF)
    # We perform multiple scattering steps (iterations)
    num_scatters = 10
    
    for _ in range(num_scatters):
        # Compute ensemble mean and covariance
        ens_mean = ensemble.mean(axis=0)
        # Use unbiased covariance
        ens_cov = np.cov(ensemble, rowvar=False, bias=False)
        
        # Innovation: y - Hx for each particle
        # Hx is ensemble[:, H_idx]
        Hx = ensemble[:, H_idx]
        innovation = observations[None, :] - Hx # (n_particles, n_obs)
        
        # Innovation covariance S = H P H^T + R
        # P is (n_state, n_state)
        # H P H^T can be computed as Cov(Hx) + R
        # Cov(Hx) is the covariance of the observed components of the ensemble
        P_H = np.cov(Hx, rowvar=False, bias=False) # (n_obs, n_obs)
        S = P_H + R_var * np.eye(n_obs)
        
        # Cross-covariance P_xH = Cov(x, Hx)
        # This is equivalent to P @ H.T
        # We can compute it directly from ensemble:
        # P_xH[i, j] = Cov(x_i, Hx_j)
        # Let's compute P (n_state x n_state) and then P_xH = P @ H.T
        # Or more efficiently, using the ensemble directly:
        # P_xH = (ensemble - ens_mean).T @ (Hx - Hx.mean(axis=0)) / (n_particles - 1)
        
        x_anom = ensemble - ens_mean
        Hx_anom = Hx - Hx.mean(axis=0)
        P_xH = (x_anom.T @ Hx_anom) / (n_particles - 1) # (n_state, n_obs)
        
        # Kalman Gain K = P_xH @ S^-1
        try:
            K = P_xH @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if S is singular
            K = P_xH @ np.linalg.pinv(S)
            
        # Update ensemble
        # x_new = x + K @ (y - Hx)
        correction = (K @ innovation.T).T # (n_particles, n_state)
        ensemble = ensemble + correction
        
        # Adaptive Inflation
        # Check spread in observed components
        ens_spread_obs = np.sqrt(np.diag(P_H))
        obs_spread = np.sqrt(np.var(observations, ddof=1) + R_var)
        # obs_spread is scalar here if obs_var is uniform? 
        # Actually obs_var might vary, but usually constant.
        # Let's use mean spread ratio
        mean_ens_spread = np.mean(ens_spread_obs)
        mean_obs_spread = np.mean(obs_spread) if np.isscalar(obs_spread) else np.mean(obs_spread)
        
        if mean_obs_spread > 0:
            spread_ratio = mean_ens_spread / mean_obs_spread
        else:
            spread_ratio = 1.0
            
        # Inflate if ensemble is too narrow
        if spread_ratio < 0.9:
            inflation_factor = 1.0 + 0.1 * (0.9 - spread_ratio)
        else:
            inflation_factor = 1.0
            
        inflation_factor = np.clip(inflation_factor, 0.95, 1.2)
        
        # Apply inflation
        new_mean = ensemble.mean(axis=0)
        ensemble = new_mean + inflation_factor * (ensemble - new_mean)

    # Final bounds check
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    ensemble = np.clip(ensemble, -100.0, 100.0)
    
    return ensemble
