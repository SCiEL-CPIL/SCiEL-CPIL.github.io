"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Global Ensemble Kalman Filter with Gaspari-Cohn Localization.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors.
    observations : ndarray, shape (10,) or (16, 10)
        Noisy observations of the observed indices.
    observed_indices : array-like
        Indices of observed coordinates (even indices: 0, 2, 4, ..., 18).
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    ensemble = np.array(ensemble, dtype=np.float64)
    observations = np.array(observations, dtype=np.float64)
    observed_indices = np.array(observed_indices)
    
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    # Normalize observations to a single vector of length n_obs
    if observations.ndim == 1:
        y = observations
    else:
        y = np.mean(observations, axis=0)
        
    # 1. Compute Background Statistics
    x_mean = np.mean(ensemble, axis=0)
    x_anom = ensemble - x_mean
    
    # Background Covariance P_b: (n_vars, n_vars)
    P_b = (x_anom.T @ x_anom) / (n_members - 1)
    
    # 2. Localization (Gaspari-Cohn)
    # Define the localization length scale. 
    # The state is 20 variables. A reasonable length scale is ~1/4 of the domain or so.
    # Let's use L = 5.0 as a starting point for the correlation length.
    L = 5.0
    
    # Compute periodic distances between all pairs of variables
    indices = np.arange(n_vars)
    dist_matrix = np.abs(indices[:, None] - indices[None, :])
    dist_matrix = np.minimum(dist_matrix, n_vars - dist_matrix)
    
    # Gaspari-Cohn function
    def gaspari_cohn(d, L):
        d = np.abs(d)
        # Piecewise polynomial
        # For d <= L/2:
        #   -5/4 (d/L)^5 + 5/2 (d/L)^4 - 5/8 (d/L)^3 - 5/3 (d/L)^2 + 1
        # For L/2 < d <= L:
        #   1/12 (d/L)^5 - 5/4 (d/L)^4 + 5/2 (d/L)^3 - 5/3 (d/L)^2 + 4/3 - 1/(6L) * d
        # Else 0
        d_L = d / L
        result = np.zeros_like(d, dtype=np.float64)
        
        mask1 = d <= 0.5 * L
        mask2 = (d > 0.5 * L) & (d <= L)
        
        # Mask 1
        if np.any(mask1):
            x = d_L[mask1]
            result[mask1] = -5/4 * x**5 + 5/2 * x**4 - 5/8 * x**3 - 5/3 * x**2 + 1
        
        # Mask 2
        if np.any(mask2):
            x = d_L[mask2]
            result[mask2] = 1/12 * x**5 - 5/4 * x**4 + 5/2 * x**3 - 5/3 * x**2 + 4/3 - 1/6 * d[mask2]
            
        return result

    L_matrix = gaspari_cohn(dist_matrix, L)
    
    # Apply localization to P_b
    P_loc = P_b * L_matrix
    
    # Ensure symmetry
    P_loc = (P_loc + P_loc.T) / 2.0
    
    # 3. Compute Kalman Gain
    # H is a linear operator: H * x -> x[observed_indices]
    # We need H @ P_loc @ H.T + R
    # H is (n_obs, n_vars)
    # Instead of building full H, we can index P_loc
    
    # P_obs = H P_loc H^T
    # P_obs[i, j] = P_loc[obs_indices[i], obs_indices[j]]
    P_obs = P_loc[np.ix_(observed_indices, observed_indices)]
    
    # R is scalar * Identity
    R = observation_variance * np.eye(n_obs)
    
    # S = P_obs + R
    S = P_obs + R
    
    # Invert S (use solve for stability, or inv for small 10x10)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        
    # K = P_loc H^T S^-1
    # K is (n_vars, n_obs)
    # K[:, j] = P_loc[:, obs_indices[j]] @ S_inv[j, :]
    # Actually, K = P_loc @ H.T @ S_inv
    # H.T is (n_vars, n_obs). H.T[:, j] is a vector with 1 at observed_indices[j].
    # So P_loc @ H.T is just selecting columns of P_loc corresponding to observed_indices.
    P_loc_Ht = P_loc[:, observed_indices] # (n_vars, n_obs)
    
    K = P_loc_Ht @ S_inv # (n_vars, n_obs)
    
    # 4. Update Ensemble
    # Innovation: y - H x_b
    # H x_b for each member: ensemble[:, observed_indices]
    H_x_b = ensemble[:, observed_indices] # (n_members, n_obs)
    innovations = y[None, :] - H_x_b # (n_members, n_obs)
    
    # x_a = x_b + K @ innovations.T ? 
    # Standard EnKF: 
    # x_i_a = x_i_b + K (y - H x_i_b)
    # K is (n_vars, n_obs). 
    # innovations is (n_members, n_obs).
    # We need to multiply K by each row of innovations.
    # Update matrix = K @ innovations.T -> (n_vars, n_members)
    # Transpose to (n_members, n_vars)
    update_matrix = (K @ innovations.T).T # (n_members, n_vars)
    
    updated_ensemble = ensemble + update_matrix
    
    # 5. Inflation
    # Multiplicative inflation around the mean
    inflation_factor = 1.05 # Slightly higher inflation for global update stability
    mean_ens = np.mean(updated_ensemble, axis=0)
    updated_ensemble = inflation_factor * (updated_ensemble - mean_ens) + mean_ens
    
    # Ensure finite values and clip
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble
# EVOLVE-BLOCK-END
