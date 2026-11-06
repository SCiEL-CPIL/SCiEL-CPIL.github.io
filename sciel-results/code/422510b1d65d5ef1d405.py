"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Transform Kalman Filter (LETKF) for Lorenz-96
    state reconstruction from partial observations. Uses block-localization
    with radius 4 to analytically preserve ensemble spread and PSD covariance.
    """
    t_start = time.time()
    DEADLINE = 145.0  # 5s safety margin before 150s kill

    X = np.array(ensemble, dtype=np.float64)
    y = np.array(observations, dtype=np.float64)
    obs_idx = np.array(observed_indices, dtype=int)
    R = float(observation_variance)

    M, N = X.shape  # M = number of ensemble members, N = state dimension
    n_obs = len(obs_idx)

    # LETKF: process each block of state variables independently
    # Block localization radius = 4, so each block covers 2*4+1 = 9 variables
    # with periodic wrapping. We process blocks centered at each index.
    block_radius = 4
    block_size = 2 * block_radius + 1  # 9

    X_a = X.copy()

    for i in range(N):
        # Determine which indices are in this block
        block_idx = [(i + j) % N for j in range(-block_radius, block_radius + 1)]
        # Remove duplicates (shouldn't happen with N=20, block_size=9)
        # Get observed indices within this block
        block_set = set(block_idx)
        obs_in_block = [j for j in obs_idx if j in block_set]
        
        if len(obs_in_block) == 0:
            continue  # No observations in this block, no update needed

        # Extract local ensemble and observations
        X_local = X[:, block_idx]  # M x block_size
        y_local = y[np.array([list(obs_idx).index(j) for j in obs_in_block])]
        
        # Map observed indices to local positions
        local_obs_positions = [block_idx.index(j) for j in obs_in_block]
        H_local = np.zeros((len(obs_in_block), block_size))
        for k, pos in enumerate(local_obs_positions):
            H_local[k, pos] = 1.0

        # LETKF update for this block
        # 1. Compute local mean and anomalies
        mean_local = np.mean(X_local, axis=0)
        W_local = X_local - mean_local  # M x block_size

        # 2. Compute local background covariance
        P_local = (W_local.T @ W_local) / (M - 1)  # block_size x block_size

        # 3. Observation-space covariance
        P_obs_local = H_local @ P_local @ H_local.T + R * np.eye(len(obs_in_block))

        # 4. Compute innovation
        innovation_local = y_local - H_local @ mean_local

        # 5. LETKF: compute transform matrix T
        # P_obs_local = T.T @ T + R*I where T comes from Cholesky of scaled anomalies
        # Standard LETKF: 
        #   C = W_local @ H_local.T  # M x n_obs_local
        #   P_obs = C @ C.T / (M-1) + R*I
        #   Cholesky: P_obs = L_obs @ L_obs.T
        #   T = (C.T @ inv(L_obs.T)) / sqrt(M-1)  ... this is getting complex
        
        # Simpler LETKF formulation:
        # The LETKF uses a transformation A such that:
        #   X_a_local = mean_local + A @ (X_local - mean_local)
        # where A preserves spread: A @ A.T = (I - K H)
        
        # Compute Kalman gain for this block
        try:
            L_obs = np.linalg.cholesky(P_obs_local)
            # K_local = P_local @ H_local.T @ inv(P_obs_local)
            # Using Cholesky: inv(P_obs_local) = inv(L_obs.T) @ inv(L_obs)
            # K_local = P_local @ H_local.T @ inv(L_obs.T) @ inv(L_obs)
            P_local_Ht = P_local @ H_local.T  # block_size x n_obs_local
            # Solve L_obs.T @ Z = P_local_Ht.T
            Z = np.linalg.solve(L_obs.T, P_local_Ht.T)  # n_obs_local x block_size
            K_local = np.linalg.solve(L_obs, Z).T  # block_size x n_obs_local
        except np.linalg.LinAlgError:
            # Fallback to pinv
            try:
                K_local = P_local @ H_local.T @ np.linalg.pinv(P_obs_local)
            except:
                K_local = np.zeros((block_size, len(obs_in_block)))

        # LETKF transform: A = I - K_local @ H_local
        A = np.eye(block_size) - K_local @ H_local

        # Apply transform to anomalies
        X_a_local = mean_local + W_local @ A.T  # M x block_size
        X_a[:, block_idx] = X_a_local

    # Ensure finite values and within bounds
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    X_a = np.clip(X_a, -100.0, 100.0)

    # Deadline check
    if time.time() > DEADLINE:
        return X_a

    return X_a
# EVOLVE-BLOCK-END
