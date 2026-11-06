"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Localized Ensemble Kalman Filter (LETKF) with adaptive spatial inflation.
    
    This method processes each state variable locally using a window of neighbors.
    It uses the Ensemble Adjustment Kalman Filter (EAKF) update to preserve the 
    ensemble mean and applies local inflation to maintain spread.
    """
    t_start = time.time()
    DEADLINE = 145.0  # Safety margin for 150s limit

    X = np.array(ensemble, dtype=np.float64)
    y = np.array(observations, dtype=np.float64)
    obs_idx = np.array(observed_indices, dtype=int)
    R = float(observation_variance)

    M, N = X.shape  # M=16, N=20
    n_obs = len(obs_idx)

    # Parameters for localization
    # Window half-width. For N=20, a window of 9 (half-width 4) covers half the domain.
    # Smaller windows reduce spurious correlations but might ignore relevant dynamics.
    # Let's try a half-width of 3 (window size 7) or 4 (window size 9).
    # Given the chaotic nature and short correlation length, 3 is often sufficient.
    half_width = 3
    
    # Precompute local observation indices for each state i
    # local_obs_indices[i] contains indices j in obs_idx such that circular_dist(i, j) <= half_width
    local_obs_indices = []
    for i in range(N):
        idxs = []
        for j in obs_idx:
            # Circular distance
            d = abs(i - j)
            d = min(d, N - d)
            if d <= half_width:
                idxs.append(j)
        local_obs_indices.append(np.array(idxs, dtype=int))

    X_a = X.copy()
    
    # We will update in-place or build a new array. 
    # Standard LETKF updates variables sequentially or in parallel?
    # Usually, it's better to update all variables based on the *original* background 
    # to avoid order-dependence, or use a sequential scan. 
    # For simplicity and symmetry, we compute updates for all i based on the background X.
    
    updates = np.zeros_like(X)
    
    for i in range(N):
        # Define local window for state i
        # Indices: i-half_width to i+half_width (circular)
        local_state_indices = [(i - half_width + k) % N for k in range(2 * half_width + 1)]
        local_state_indices = np.array(local_state_indices, dtype=int)
        L = len(local_state_indices)
        
        # Local observations
        local_obs_idx = local_obs_indices[i]
        n_local_obs = len(local_obs_idx)
        
        if n_local_obs == 0:
            # No observations in this window, do not update (or just inflate? 
            # Usually, if no obs, we leave it or apply a small global inflation. 
            # Let's leave it unchanged for now, as it's unobserved in the local sense.)
            continue
            
        # Extract local background ensemble
        # Shape: (M, L)
        X_local = X[:, local_state_indices]
        
        # Local mean
        mean_local = np.mean(X_local, axis=0)
        
        # Local anomalies
        W_local = X_local - mean_local  # (M, L)
        
        # Local covariance P_b (L x L)
        P_b = (W_local.T @ W_local) / (M - 1)
        
        # Observation operator for this local block:
        # We need to map the local state indices to the global observation indices.
        # The observations are at global indices local_obs_idx.
        # The local state indices are local_state_indices.
        # We need to find which local state components are observed.
        
        # Create a mapping from local state index to global index
        # local_state_indices[k] is the global index of the k-th local state.
        # local_obs_idx[m] is the global index of the m-th local observation.
        
        # We need H_local: (n_local_obs x L)
        # H_local[m, k] = 1 if local_state_indices[k] == local_obs_idx[m]
        
        H_local = np.zeros((n_local_obs, L))
        for m, obs_g_idx in enumerate(local_obs_idx):
            # Find if obs_g_idx is in local_state_indices
            match = np.where(local_state_indices == obs_g_idx)[0]
            if len(match) > 0:
                H_local[m, match[0]] = 1.0
        
        # Observation vector
        y_local = y[local_obs_idx]  # (n_local_obs,)
        
        # Innovation
        innovation = y_local - (H_local @ mean_local)
        
        # Compute P_obs = H P_b H^T + R I
        HPb = H_local @ P_b
        P_obs = HPb @ H_local.T + R * np.eye(n_local_obs)
        
        # Compute Kalman Gain K = P_b H^T P_obs^-1
        # K shape: (L, n_local_obs)
        # P_b H^T shape: (L, n_local_obs)
        Pb_Ht = P_b @ H_local.T
        
        try:
            # Solve P_obs K^T = Pb_Ht  => K = (P_obs^-1 Pb_Ht)^T
            # Using Cholesky for stability
            L_obs = np.linalg.cholesky(P_obs)
            # Solve L_obs Z = Pb_Ht.T
            Z = np.linalg.solve(L_obs, Pb_Ht.T)
            # Solve L_obs.T K.T = Z
            K_t = np.linalg.solve(L_obs.T, Z)
            K = K_t.T
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if Cholesky fails
            K = Pb_Ht @ np.linalg.pinv(P_obs)

        # EAKF Update for the local window
        # X_a_local = mean_local + W_local @ (I - K H)^T ? No, that's for standard EnKF.
        # EAKF formula:
        # X_a_m = mean + (X_m - mean) @ (I - K H)^T  is incorrect for EAKF.
        
        # Correct EAKF update:
        # Let A = I - K H. (n_obs x n_obs matrix? No, H is n_obs x L, K is L x n_obs. KH is n_obs x n_obs)
        # The update for each member m:
        # x_a_m = x_mean + (x_m - x_mean) @ (I - K H)^T  ... wait, dimensions.
        # x_m is (L,). x_mean is (L,).
        # The term (x_m - x_mean) is a row vector (1, L).
        # We need to multiply by a (L x L) matrix?
        
        # The EAKF update is:
        # x_a_m = x_mean + (x_m - x_mean) @ (I - K H)^T is not right because (I-KH) is n_obs x n_obs.
        
        # Let's use the standard EAKF formula derived from the ensemble covariance update.
        # P_a = (I - K H) P_b (I - K H)^T + K R K^T
        # The ensemble anomalies update is:
        # W_a = W_b @ (I - K H)^T ? No.
        
        # Actually, a simpler way to implement EAKF is:
        # 1. Compute the Kalman gain K.
        # 2. Compute the matrix M = I - K H (where H is the local obs operator, n_obs x L).
        #    Wait, K is L x n_obs. H is n_obs x L. KH is n_obs x n_obs.
        #    This doesn't apply to the state anomalies directly as a simple matrix mult.
        
        # Let's stick to the **Standard EnKF** update but applied locally.
        # x_a_m = x_b_m + K (y - H x_b_m)
        # This is the most straightforward and robust.
        
        H_x_local = H_local @ X_local  # (n_local_obs, M)
        innovations_m = y_local[:, None] - H_x_local  # (n_local_obs, M)
        
        # Update: x_a_m = x_b_m + K @ innovations_m
        # K is (L, n_local_obs). innovations_m is (n_local_obs, M).
        # Result is (L, M).
        X_a_local = X_local + K @ innovations_m
        
        # Ensure the mean is preserved exactly
        mean_a_local = np.mean(X_a_local, axis=1)
        mean_b_local = mean_local
        X_a_local = X_a_local + (mean_b_local - mean_a_local)[:, None]
        
        # Local Inflation
        # Calculate innovation variance
        obs_var = np.var(y_local)
        # Expected innovation variance
        ens_var_obs = np.var(X_local[:, H_local.nonzero()[1]], axis=0) if H_local.any() else 0
        # More accurately:
        # The variance of the observation from the ensemble is var(H X_b)
        # H X_b is (n_obs, M).
        # var(H X_b) = np.var(H_local @ X_local, axis=1)
        pred_obs_var = np.var(H_local @ X_local, axis=1)
        expected_innov_var = np.mean(pred_obs_var + R)
        
        if expected_innov_var > 1e-6:
            inflation_factor = np.sqrt(obs_var / expected_innov_var)
            # Clip inflation to prevent explosion
            inflation_factor = np.clip(inflation_factor, 1.0, 1.2)
        else:
            inflation_factor = 1.0
            
        # Apply inflation to the local ensemble around the local mean
        X_a_local = mean_b_local + inflation_factor * (X_a_local - mean_b_local)
        
        # Store the updated local window back into the global array
        # Note: Since we are updating based on the original background X, we can just assign.
        # If there is overlap between windows, the later assignment will overwrite.
        # To avoid order dependence, we could average, but standard LETKF often just
        # updates the center or uses a specific scheme.
        # However, since we are updating *all* i, and each i is the center of its window,
        # the value at i is updated by the window centered at i.
        # The neighbors are also updated by their own windows.
        # This is a "parallel" local update.
        
        X_a[:, local_state_indices] = X_a_local.T

    # Global sanity check and inflation if local ones failed or to ensure global spread
    # Check global innovation
    global_innovation = y - X[:, obs_idx].mean(axis=0)
    global_innov_var = np.var(global_innovation)
    global_pred_var = np.mean(np.var(X[:, obs_idx], axis=0) + R)
    
    if global_pred_var > 1e-6:
        global_inflation = np.sqrt(global_innov_var / global_pred_var)
        global_inflation = np.clip(global_inflation, 1.0, 1.1) # Gentler global correction
    else:
        global_inflation = 1.0
        
    # Apply global inflation if local inflation was insufficient?
    # Or just apply a very small global inflation to ensure we don't collapse globally.
    # Let's blend: if global inflation is significantly higher, apply it.
    # Actually, the local inflation should handle it.
    # We'll just ensure finiteness.
    
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    X_a = np.clip(X_a, -100.0, 100.0)

    if time.time() > DEADLINE:
        return X_a

    return X_a
# EVOLVE-BLOCK-END
