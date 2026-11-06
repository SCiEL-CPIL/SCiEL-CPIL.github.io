"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_start_time = None

def _get_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Localized Ensemble Kalman Filter (EnKF) with adaptive inflation.
    Uses block localization to handle the 20-variable Lorenz-96 system.
    """
    M, N = ensemble.shape
    obs_var = observation_variance
    obs_idx = np.asarray(observed_indices, dtype=int)
    n_obs = len(obs_idx)
    
    # Deadline guard
    elapsed = _get_time()
    if elapsed > 130.0:
        result = np.array(ensemble, dtype=np.float64).copy()
        obs_arr = np.asarray(observations, dtype=np.float64)
        if obs_arr.ndim == 1:
            obs_arr = np.tile(obs_arr, (M, 1))
        result[:, obs_idx] += 0.5 * (obs_arr - result[:, obs_idx])
        np.clip(result, -100.0, 100.0, out=result)
        return result
    
    result = np.array(ensemble, dtype=np.float64).copy()
    
    # Prepare observation array
    obs_arr = np.asarray(observations, dtype=np.float64)
    if obs_arr.ndim == 1:
        obs_arr = np.tile(obs_arr, (M, 1))
    
    # Block localization radius - use smaller blocks for better conditioning
    R = 2
    
    # Build block structure: each observed index gets a block of radius R
    # Merge overlapping blocks
    in_block = np.zeros(N, dtype=bool)
    for j in obs_idx:
        for d in range(-R, R + 1):
            idx = (j + d) % N
            in_block[idx] = True
    
    # Assign block IDs using connected components
    block_ids = np.full(N, -1, dtype=int)
    block_count = 0
    assigned = np.zeros(N, dtype=bool)
    
    for j in obs_idx:
        if assigned[j]:
            continue
        
        # BFS to find connected component
        queue = [j]
        assigned[j] = True
        block_members = []
        
        while queue:
            current = queue.pop(0)
            block_members.append(current)
            for d in range(-R, R + 1):
                neighbor = (current + d) % N
                if in_block[neighbor] and not assigned[neighbor]:
                    assigned[neighbor] = True
                    queue.append(neighbor)
        
        for member in block_members:
            block_ids[member] = block_count
        block_count += 1
    
    # Process each block independently with localized EnKF
    for b in range(block_count):
        members = np.where(block_ids == b)[0]
        if len(members) == 0:
            continue
        
        # Observed indices within this block
        obs_in_block = [j for j in obs_idx if block_ids[j] == b]
        if len(obs_in_block) == 0:
            continue
        obs_in_block = np.array(obs_in_block, dtype=int)
        n_obs_block = len(obs_in_block)
        
        # Extract ensemble for this block
        X_block = result[:, members]  # (M, n_members)
        X_obs = result[:, obs_in_block]  # (M, n_obs_block)
        
        # Observation values for this block
        obs_local_vals = np.zeros((M, n_obs_block))
        for k, j in enumerate(obs_in_block):
            pos = np.where(obs_idx == j)[0]
            if len(pos) > 0:
                obs_local_vals[:, k] = obs_arr[:, pos[0]]
        
        # Ensemble mean and anomalies
        X_mean = np.mean(X_block, axis=0, keepdims=True)
        X_anom = X_block - X_mean  # (M, n_members)
        X_obs_mean = np.mean(X_obs, axis=0, keepdims=True)
        X_obs_anom = X_obs - X_obs_mean  # (M, n_obs_block)
        
        # Innovation
        innovation = obs_local_vals - X_obs  # (M, n_obs_block)
        
        # Cross-covariance: P_xo = (1/(M-1)) * X_anom.T @ X_obs_anom
        cross_cov = (X_anom.T @ X_obs_anom) / (M - 1)  # (n_members, n_obs_block)
        
        # Innovation covariance: P_oo + R
        innov_cov = (X_obs_anom.T @ X_obs_anom) / (M - 1) + obs_var * np.eye(n_obs_block)
        
        # Regularize for numerical stability
        innov_cov += 1e-8 * np.eye(n_obs_block)
        
        try:
            innov_cov_inv = np.linalg.inv(innov_cov)
        except np.linalg.LinAlgError:
            innov_cov_inv = np.linalg.pinv(innov_cov)
        
        # Kalman gain
        K = cross_cov @ innov_cov_inv  # (n_members, n_obs_block)
        
        # Update: x_a = x_f + K @ (y - H x_f)
        # For each member: x_a_m = x_f_m + K @ (y_m - H x_f_m)
        update = K @ innovation.T  # (n_members, M)
        X_block_new = X_block + update.T  # (M, n_members)
        
        # Apply adaptive inflation to prevent ensemble collapse
        # Compute spread ratio
        spread_ratio = np.mean(np.var(X_block_new, axis=0)) / (np.mean(np.var(X_block, axis=0)) + 1e-10)
        
        # If spread decreased too much, inflate
        target_spread = 1.0
        if spread_ratio < target_spread:
            inflation_factor = 1.0 + 0.1 * (target_spread - spread_ratio)
            inflation_factor = np.clip(inflation_factor, 1.0, 1.5)
            
            # Inflate anomalies around the mean
            X_block_new_mean = np.mean(X_block_new, axis=0, keepdims=True)
            X_block_new_anom = X_block_new - X_block_new_mean
            X_block_new = X_block_new_mean + inflation_factor * X_block_new_anom
        
        # Apply to result
        result[:, members] = X_block_new
    
    # Final safety checks
    np.clip(result, -100.0, 100.0, out=result)
    
    if not np.all(np.isfinite(result)):
        result = np.array(ensemble, dtype=np.float64).copy()
        result[:, obs_idx] += 0.5 * (obs_arr - result[:, obs_idx])
        np.clip(result, -100.0, 100.0, out=result)
    
    return result

# EVOLVE-BLOCK-END
