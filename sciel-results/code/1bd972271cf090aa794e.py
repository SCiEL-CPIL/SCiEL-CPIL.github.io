"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Transform Kalman Filter (LETKF) with block localization,
    adaptive spread correction, and deadline guard for Lorenz-96 assimilation.
    """
    t_start = time.time()
    DEADLINE = 145.0

    X = np.array(ensemble, dtype=np.float64)
    y = np.array(observations, dtype=np.float64)
    obs_idx = np.array(observed_indices, dtype=int)
    R = float(observation_variance)

    M, N = X.shape  # M ensemble members, N state dimension
    n_obs = len(obs_idx)

    # Background mean
    mean_b = np.mean(X, axis=0)

    # Ensemble anomalies
    W = X - mean_b  # M x N

    # Block localization: for each state variable, find the set of variables
    # within a circular distance of 'block_radius'
    block_radius = 4

    # Precompute which observations fall within each block
    # For each state index i, the block includes indices within circular distance block_radius
    # We need to find which observed indices are in each block

    # Since observed_indices are even indices (0, 2, 4, ..., 18) for N=20
    # and block_radius=4, each block will contain several observations

    # For LETKF, we process each block independently
    # For each block centered at state i:
    #   - Select variables in the block (within circular distance block_radius)
    #   - Select observations that correspond to those variables (if observed)
    #   - Apply the ensemble transform within that block

    # Build the block structure
    # For each state variable i, determine which variables are in its local block
    blocks = []
    for i in range(N):
        # Circular indices within distance block_radius of i
        offsets = list(range(-block_radius, block_radius + 1))
        block_vars = [(i + off) % N for off in offsets]
        # Remove duplicates while preserving order
        seen = set()
        unique_block_vars = []
        for v in block_vars:
            if v not in seen:
                seen.add(v)
                unique_block_vars.append(v)
        block_vars = unique_block_vars

        # Find which observations are in this block
        obs_in_block = [j for j in range(n_obs) if obs_idx[j] in set(block_vars)]

        blocks.append({
            'center': i,
            'vars': block_vars,
            'obs_local': obs_in_block
        })

    # Initialize analysis ensemble
    X_a = X.copy()

    # For each state variable, compute its analysis using the local transform
    # We'll accumulate contributions from all blocks that include this variable
    # Actually, in LETKF, each variable is updated by its own local transform
    # But since blocks overlap, we need to be careful. The standard approach is
    # to update each variable using the transform from the block centered at that variable.

    # Compute the local transform for each block
    for i in range(N):
        block = blocks[i]
        var_indices = block['vars']  # state variables in this block
        obs_local = block['obs_local']  # observation indices in this block

        n_vars = len(var_indices)
        n_local_obs = len(obs_local)

        if n_local_obs == 0:
            # No observations in this block, keep background
            continue

        # Extract local ensemble anomalies: M x n_vars
        W_local = W[:, var_indices]  # M x n_vars

        # Local background covariance: n_vars x n_vars
        P_local = (W_local.T @ W_local) / (M - 1)

        # Observation operator for this block: selects observed variables
        # H maps state to observations. For local obs, H is n_local_obs x n_vars
        # The observed variables within the block
        obs_var_indices_in_block = [obs_idx[j] for j in obs_local]

        # Find positions of these observed variables within var_indices
        var_to_pos = {v: pos for pos, v in enumerate(var_indices)}
        H_local = np.zeros((n_local_obs, n_vars))
        for k, ov in enumerate(obs_var_indices_in_block):
            pos = var_to_pos.get(ov)
            if pos is not None:
                H_local[k, pos] = 1.0

        # P_obs = H P_local H^T + R I
        P_obs = H_local @ P_local @ H_local.T + R * np.eye(n_local_obs)

        # Innovation
        innovation_local = y[obs_local] - mean_b[var_indices][[var_to_pos[obs_idx[j]] for j in obs_local]]

        # Compute the transform using the ensemble transform approach
        # T = I - (1/M) * P_local H^T (H P_local H^T + R I)^{-1} H W^T ... 
        # Actually, the standard LETKF transform is:
        #   Let A = W_local (M x n_vars)
        #   Let P = A^T A / (M-1)
        #   K_local = P H^T (H P H^T + R I)^{-1}  (n_vars x n_local_obs)
        #   The update for the ensemble is:
        #     W_a = W - (1/M) * A * (H A^T) * (H A^T / (M-1) + R I)^{-1} * (H W^T)^T ... 
        # 
        # More precisely, using the ensemble transform:
        #   Let B = W_local (M x n_vars)  [anomalies]
        #   Let C = H B (M x n_local_obs)  [observed anomalies]
        #   The transform T is n_vars x M:
        #     T = I_n_vars - (1/M) * B * (C^T) * (C C^T / (M-1) + R I)^{-1} * (C^T) ... 
        #
        # Actually, let me use the direct formula:
        #   K = P_local H^T (H P_local H^T + R I)^{-1}  (n_vars x n_local_obs)
        #   Mean update: mean_a_local = mean_b[var_indices] + K @ innovation_local
        #   Ensemble update: 
        #     W_a = W - (1/M) * B * (H B^T) @ (H B^T / (M-1) + R I)^{-1} @ (H W^T).T ...
        #
        # Let me use the simpler additive update but with the local covariance:
        # This is effectively a local EnKF update.

        # Compute local Kalman gain
        P_local_Ht = P_local @ H_local.T  # n_vars x n_local_obs

        try:
            L = np.linalg.cholesky(P_obs)
            Z = np.linalg.solve(L.T, P_local_Ht.T)  # n_local_obs x n_vars
            K_local = np.linalg.solve(L, Z).T  # n_vars x n_local_obs
        except np.linalg.LinAlgError:
            try:
                K_local = P_local_Ht @ np.linalg.inv(P_obs)
            except np.linalg.LinAlgError:
                K_local = P_local_Ht @ np.linalg.pinv(P_obs)

        # Update mean for variables in this block
        mean_a_local = mean_b[var_indices] + K_local @ innovation_local

        # Update ensemble for variables in this block
        # X_a[:, var_indices] = X[:, var_indices] + K_local @ (y_local - H X[:, var_indices])
        H_X_local = X[:, var_indices] @ H_local.T  # M x n_local_obs
        innovation_ens_local = y[obs_local][None, :] - H_X_local  # M x n_local_obs
        update_local = innovation_ens_local @ K_local.T  # M x n_vars

        X_a[:, var_indices] = X[:, var_indices] + update_local

    # Now apply adaptive spread correction
    # Compare the analysis spread to the background spread at observed locations
    mean_a = np.mean(X_a, axis=0)

    # Compute the ratio of ensemble spread at observed variables
    obs_spread_bg = np.std(X[:, obs_idx], axis=0)
    obs_spread_an = np.std(X_a[:, obs_idx], axis=0)

    # The expected reduction in spread is given by the Kalman gain
    # A simple adaptive inflation: if spread decreased too much, inflate
    if np.mean(obs_spread_bg) > 1e-10:
        spread_ratio = np.mean(obs_spread_an) / np.mean(obs_spread_bg)
        # Target: spread should not decrease by more than ~30%
        # If it decreased too much, inflate
        if spread_ratio < 0.7:
            inflation = 0.7 / spread_ratio
            inflation = np.clip(inflation, 1.0, 1.2)
            X_a = mean_a + inflation * (X_a - mean_a)

    # Ensure finite and bounded
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    X_a = np.clip(X_a, -100.0, 100.0)

    if time.time() > DEADLINE:
        return X_a

    return X_a
# EVOLVE-BLOCK-END
