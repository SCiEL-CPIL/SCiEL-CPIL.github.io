"""Local Ensemble Transform Kalman Filter (LETKF) for Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    LETKF analysis update for Lorenz-96.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of even-indexed states
    - observed_indices: array of observed indices (even indices)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble with same (16, 20) shape
    """
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin before 150s kill
    
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=int)
    
    nm, nvar = ensemble.shape  # 16, 20
    nobs = len(observations)   # 10
    
    # Block localization radius
    block_radius = 4
    
    # Build blocks: each observed variable and its neighbors within block_radius
    # For each observed index, find all state indices within block_radius
    # Then process each block independently
    
    result = ensemble.copy()
    
    # For each observed variable, define a block of state variables
    # The block includes the observed variable and its nearest neighbors
    # within block_radius in the periodic state space
    
    for i_obs in range(nobs):
        obs_idx = observed_indices[i_obs]
        
        # Find state indices within block_radius of obs_idx (periodic)
        block_indices = []
        for offset in range(-block_radius, block_radius + 1):
            idx = (obs_idx + offset) % nvar
            if idx not in block_indices:
                block_indices.append(idx)
        
        block_indices = np.array(block_indices)
        nblock = len(block_indices)
        
        # Extract the block: ensemble members x block variables
        # Shape: (nm, nblock)
        block_ens = ensemble[:, block_indices]
        
        # The observation corresponds to the first variable in the block
        # (since obs_idx is the center, but let's find its position)
        obs_pos_in_block = np.where(block_indices == obs_idx)[0][0]
        
        # Compute the observation operator for this block:
        # y = H x, where H picks out the obs_pos_in_block variable
        # H is (1, nblock) with 1 at obs_pos_in_block
        
        # LETKF: transform the ensemble in the block
        # 1. Compute ensemble mean and anomaly matrix
        mean_block = np.mean(block_ens, axis=0)  # (nblock,)
        anomalies = block_ens - mean_block  # (nm, nblock)
        
        # 2. Compute the innovation
        # y_obs - H * mean_ensemble
        y_obs_scalar = observations[i_obs]
        y_mean = mean_block[obs_pos_in_block]
        innovation = y_obs_scalar - y_mean  # scalar
        
        # 3. Compute the observation-space covariance
        # P_yy = H P H^T + R
        # H P H^T = (1/nm) * sum of (anomaly[obs_pos])^2
        obs_anomalies = anomalies[:, obs_pos_in_block]  # (nm,)
        P_yy = np.mean(obs_anomalies ** 2) + observation_variance
        
        # 4. Compute the Kalman gain in the local block
        # For LETKF, we need to compute the transformation matrix
        # Y = H A / sqrt(nm-1) where A is the anomaly matrix
        # Y is (nobs_block, nm-1)
        
        # For a single observation per block:
        # Y = obs_anomalies / sqrt(nm-1)  (1, nm-1)
        # Y^T Y / (nm-1) = P_yy_ens / (nm-1) where P_yy_ens = sum obs_anom^2
        # Actually, let's use the standard LETKF formulation
        
        # The LETKF transformation:
        # Let Y = H A / sqrt(nm-1), shape (1, nm-1) for one obs
        # Compute M = (I + Y^T Y / (nm-1))^{-1/2}
        # But we need to be careful with the normalization
        
        # Standard LETKF:
        # A = anomalies (nm, nblock)
        # Y = H A / sqrt(nm-1) (nobs_block, nm-1)
        # For nobs_block=1: Y is (1, nm-1)
        # Y^T Y is (nm-1, nm-1)
        # Compute the transformation using the fact that we only need
        # to update in the direction of the observation
        
        # Simplified: for one observation per block, the LETKF update
        # is equivalent to a scalar Kalman filter applied to the 
        # ensemble mean and a rotation of the anomalies
        
        # The updated ensemble mean:
        # K = P_xy / P_yy where P_xy = H P (scalar here)
        # P_xy = mean of obs_anomalies * (block_anomalies) / nm... 
        # Actually: P_xy = (1/nm) * A^T Y_obs... 
        
        # Let me use the proper LETKF formulation:
        # A = (nm, nblock) anomaly matrix
        # Y = H A / sqrt(nm-1)  (1, nm-1)
        # y_vec = innovation / sqrt(P_yy)  (1,)
        
        # The transformation:
        # M = (I_{nm-1} + Y^T Y / (nm-1))^{-1/2}
        # But Y is (1, nm-1), so Y^T Y is (nm-1, nm-1)
        
        # For computational efficiency with small blocks, let's do it directly:
        Y = obs_anomalies / np.sqrt(nm - 1)  # (nm-1,) -- wait, obs_anomalies is (nm,)
        
        # Actually, we need to use only nm-1 anomalies for the LETKF
        # Let's use the first nm-1 rows
        obs_anoms_reduced = obs_anomalies[:nm-1]  # (nm-1,)
        A_reduced = anomalies[:nm-1, :]  # (nm-1, nblock)
        
        Y = obs_anoms_reduced / np.sqrt(nm - 1)  # (nm-1,)
        
        # Y^T Y is a scalar here since Y is (nm-1,) treated as column
        # Actually Y is (1, nm-1) in matrix form, so Y^T Y is (nm-1, nm-1)
        # Y^T Y = outer(Y, Y)  (nm-1, nm-1)
        
        YTY = np.outer(Y, Y)  # (nm-1, nm-1)
        
        # Compute M = (I + YTY/(nm-1))^{-1/2}
        # For a rank-1 update, we can use the Sherman-Morrison formula
        # I + alpha * v v^T where alpha = 1/(nm-1), v = Y
        # (I + alpha v v^T)^{-1} = I - alpha v v^T / (1 + alpha v^T v)
        
        alpha = 1.0 / (nm - 1)
        v = Y  # (nm-1,)
        vTv = np.dot(v, v)  # scalar
        denom = 1.0 + alpha * vTv
        
        # (I + alpha v v^T)^{-1/2} can be computed via eigenvalue decomposition
        # or we can use the fact that for rank-1 updates:
        # (I + alpha v v^T)^{-1/2} = I - (alpha / (1 + alpha vTv)) * v v^T * (1 - sqrt(1/(1+alpha vTv)))
        # This is getting complex. Let's just do eigendecomposition for small nm.
        
        # Actually, for nm=16, nm-1=15, this is small enough
        M_matrix = np.eye(nm - 1) + YTY
        # Compute M^{-1/2}
        eigvals, eigvecs = np.linalg.eigh(M_matrix)
        eigvals_inv_sqrt = 1.0 / np.sqrt(eigvals)
        M_inv_sqrt = eigvecs @ np.diag(eigvals_inv_sqrt) @ eigvecs.T
        
        # The updated anomalies:
        # A_new = M * A_reduced  (nm-1, nblock)
        A_new = M_inv_sqrt @ A_reduced  # (nm-1, nblock)
        
        # The updated mean:
        # mean_new = mean + K * innovation
        # K = P_xy / P_yy
        # P_xy = (1/(nm-1)) * sum of obs_anoms_reduced * A_reduced[:, obs_pos_in_block]
        # Wait, P_xy for the full block:
        # P_xy = H P = (1/nm) * A^T obs_anomalies  (nblock,)
        
        P_xy = np.mean(anomalies * obs_anomalies[:, np.newaxis], axis=0)  # (nblock,)
        K = P_xy / P_yy  # (nblock,)
        
        mean_new = mean_block + K * innovation  # (nblock,)
        
        # Construct the new ensemble:
        # The new ensemble members are:
        # member_j = mean_new + A_new[j, :] for j=0,...,nm-2
        # member_{nm-1} is determined by the constraint that the mean is preserved
        
        new_ens_block = np.zeros((nm, nblock))
        for j in range(nm - 1):
            new_ens_block[j] = mean_new + A_new[j, :]
        
        # The last member:
        # sum of all members = nm * mean_new
        # sum of first nm-1 members = (nm-1) * mean_new + sum(A_new)
        # So last member = nm * mean_new - sum(first nm-1)
        sum_first = np.sum(new_ens_block[:nm-1], axis=0)
        new_ens_block[nm-1] = nm * mean_new - sum_first
        
        # Update the result
        result[:, block_indices] = new_ens_block
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # Clip to valid range
    result = np.clip(result, -100.0, 100.0)
    
    # Ensure finite values
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return result
# EVOLVE-BLOCK-END
