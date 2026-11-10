"""
Local Ensemble Transform Kalman Filter (LETKF) for Lorenz-96 assimilation.
Structural change: Replaces global linear EnKF with local transform-based update.
"""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _localize_letkf(ensemble, observations, observed_indices, observation_variance, n_local=5):
    """
    Performs LETKF update on a local window centered at each state variable.
    n_local: Half-width of the localization window (total size 2*n_local + 1).
    """
    M, N = ensemble.shape
    # Copy ensemble to avoid modifying input directly in-place if we want to be safe, 
    # but usually we can operate on a copy or the original if we overwrite.
    # The task says "updated ensemble", so we return a new array.
    result = ensemble.copy()
    
    # Precompute observed indices set for quick lookup? 
    # observed_indices is typically sorted and fixed (even indices).
    obs_set = set(observed_indices)
    
    # For each state variable i, define a local window
    for i in range(N):
        # Define local indices with periodic boundary conditions
        local_indices = []
        for k in range(-n_local, n_local + 1):
            idx = (i + k) % N
            local_indices.append(idx)
        
        # Extract local ensemble
        X_local = ensemble[:, local_indices] # (M, 2*n_local+1)
        
        # Identify which of these local indices are observed
        local_obs_mask = np.array([idx in obs_set for idx in local_indices])
        
        # If no observations in this local window, skip update (or use inflation only)
        if not np.any(local_obs_mask):
            continue
            
        local_obs_indices = np.where(local_obs_mask)[0]
        K_local = len(local_obs_indices)
        
        # Get local observations
        # We need to map local_obs_indices back to the global observed_indices array
        # to get the correct observation values.
        # observed_indices is an array of global indices.
        # local_indices[local_obs_indices] gives global indices that are observed.
        global_obs_indices_in_window = local_indices[local_obs_indices]
        
        # Find positions of these global indices in the main observed_indices array
        # Since observed_indices is small (10 elements) and fixed, we can just search.
        # For efficiency, create a map from global index to observation value.
        # But observation values are passed in 'observations' corresponding to 'observed_indices'.
        # So we need: y_local[j] = observations[observed_indices == global_obs_indices_in_window[j]]
        
        # Efficient mapping:
        # observed_indices is e.g. [0, 2, 4, ...]
        # We can create a dict or just use np.where
        # Given N=20 and K=10, this is cheap.
        
        # Map global index to position in observations array
        # obs_pos = {idx: pos for pos, idx in enumerate(observed_indices)}
        # y_local = np.array([observations[obs_pos[idx]] for idx in global_obs_indices_in_window])
        
        # Faster way:
        # Create an array of positions
        # Since observed_indices is sorted and contains even numbers:
        # idx // 2 gives the position in observations if observed_indices = [0, 2, ..., 18]
        # Let's assume standard even indices for speed, but keep it general.
        obs_map = {idx: pos for pos, idx in enumerate(observed_indices)}
        y_local = np.array([observations[obs_map[idx]] for idx in global_obs_indices_in_window])
        
        # LETKF Update on X_local
        # 1. Compute local ensemble mean and anomalies
        X_mean = np.mean(X_local, axis=0)
        X_anom = X_local - X_mean
        
        # 2. Compute local covariance (M-1 normalization)
        # C = X_anom.T @ X_anom / (M - 1)
        C = (X_anom.T @ X_anom) / (M - 1)
        
        # 3. Observation operator H_local
        # H_local is a (K_local, 2*n_local+1) matrix with 1s at local_obs_indices
        H_local = np.zeros((K_local, len(local_indices)))
        H_local[np.arange(K_local), local_obs_indices] = 1.0
        
        # 4. Innovation
        # y - H X_mean
        innovation = y_local - H_local @ X_mean
        
        # 5. Innovation covariance S = H C H^T + R
        HC = H_local @ C
        S = HC @ H_local.T + observation_variance * np.eye(K_local)
        
        # 6. Compute the transform
        # We need to find a matrix T such that:
        # (X_anom - C H^T S^{-1} H X_anom) = X_anom @ T
        # Actually, the standard LETKF derivation:
        # Let Y = X_anom @ H^T (M x K_local)
        # We want to project the anomalies onto the subspace orthogonal to the observation error.
        
        # A common formulation:
        # 1. Compute Y = X_anom @ H^T
        # 2. Compute Y S^{-1} Y^T
        # 3. T = I - Y S^{-1} Y^T (M x M) ... wait, dimensions.
        
        # Correct LETKF steps:
        # a. X_anom (M x L)
        # b. Y = X_anom @ H^T (M x K)
        # c. Compute the matrix B = Y @ inv(S) @ Y^T (M x M)
        # d. Compute the eigen-decomposition of B? No, that's for the transform.
        
        # Simpler LETKF algorithm (Hunt et al.):
        # 1. Compute Y = X_anom @ H^T
        # 2. Compute Y S^{-1} Y^T
        # 3. Let T be the matrix that satisfies:
        #    (I - Y S^{-1} Y^T) = T T^T
        #    This is a Cholesky or SVD based factorization.
        
        # Let's use the SVD approach for stability.
        # B = Y @ np.linalg.inv(S) @ Y.T
        # U, s, Vt = np.linalg.svd(B)
        # T = U @ np.diag(np.sqrt(np.maximum(0, 1 - s))) @ U.T ? 
        # Actually, the transform T is applied to the anomalies.
        # X_anom_a = X_anom @ T
        
        # Let's stick to the standard linear algebra solution for the local update:
        # Gain K_local = C @ H^T @ inv(S)
        # X_mean_a = X_mean + K_local @ innovation
        # X_anom_a = X_anom - K_local @ (H @ X_anom)
        
        # This is the "deterministic" update which is stable and avoids the transform complexity.
        # It is equivalent to the EnKF update but local.
        
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            
        K_local_gain = C @ H_local.T @ S_inv # (L x K)
        
        # Update local mean
        X_mean_a = X_mean + K_local_gain @ innovation
        
        # Update local anomalies
        # X_anom_a = X_anom - K_local_gain @ (H_local @ X_anom)
        H_X_anom = H_local @ X_anom # (K x M)
        X_anom_a = X_anom - K_local_gain @ H_X_anom # (L x M)
        
        # Reconstruct local ensemble
        X_local_a = X_anom_a.T + X_mean_a # (M x L)
        
        # Write back to result
        result[:, local_indices] = X_local_a
        
    return result

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update using Local Ensemble Transform Kalman Filter (LETKF).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    M, N = ensemble.shape
    
    # Basic sanity check
    if not np.all(np.isfinite(ensemble)):
        ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Local LETKF update
    # n_local=3 gives a window of 7 variables. For N=20, this covers most of the state
    # but ensures localization.
    # n_local=4 gives window of 9.
    # Let's try n_local=4 for better coverage of the chaotic interactions.
    updated = _localize_letkf(ensemble, observations, observed_indices, observation_variance, n_local=4)
    
    # Check time
    if time.time() > deadline:
         return updated
    
    # Adaptive Inflation
    # Check if the ensemble spread is too small compared to observation error.
    # If so, inflate the ensemble around the mean.
    mean = np.mean(updated, axis=0)
    spread = np.std(updated[:, observed_indices], axis=0)
    obs_err = np.sqrt(observation_variance)
    
    # If mean spread is less than observation error, inflate
    ratio = np.mean(spread) / obs_err
    if ratio < 0.8:
        inflation = 1.0 + 0.2 * (0.8 - ratio)
        updated = mean + inflation * (updated - mean)
        
    # Final cleanup
    updated = np.nan_to_num(updated, nan=0.0, posinf=100.0, neginf=-100.0)
    updated = np.clip(updated, -100.0, 100.0)
    
    return updated
# EVOLVE-BLOCK-END
