import numpy as np
import time


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Adjustment Kalman Filter (EnAKF) for Lorenz-96.
    
    Parameters:
    ensemble: (M, N) array of forecast ensemble states
    observations: (K,) array of noisy observations
    observed_indices: array of indices that are observed
    observation_variance: scalar observation noise variance
    
    Returns:
    updated ensemble, same shape (M, N)
    """
    t0 = time.time()
    deadline = t0 + 115.0

    M, N = ensemble.shape
    K = len(observed_indices)

    # Build observation lookup
    obs_dict = {}
    for i, idx in enumerate(observed_indices):
        obs_dict[idx] = observations[i]

    # --- Adaptive multiplicative inflation ---
    if M > 1:
        obs_spread = np.std(ensemble[:, observed_indices], axis=0)
        obs_err = np.sqrt(observation_variance)
        ratio = np.mean(obs_spread) / obs_err if obs_err > 0 else 1.0
        if ratio < 1.0:
            infl = 1.0 + 0.08 * (1.0 - ratio)
            infl = np.clip(infl, 1.01, 1.06)
        else:
            infl = 1.01
        mean = np.mean(ensemble, axis=0)
        ensemble = mean + infl * (ensemble - mean)

    result = ensemble.copy()
    window_half = 2
    window_size = 2 * window_half + 1

    for i in range(N):
        if time.time() > deadline:
            break

        local_indices = [(i + j) % N for j in range(-window_half, window_half + 1)]

        local_obs_mask = np.zeros(window_size, dtype=bool)
        local_obs_vals = np.zeros(window_size)
        for l_idx, g_idx in enumerate(local_indices):
            if g_idx in obs_dict:
                local_obs_mask[l_idx] = True
                local_obs_vals[l_idx] = obs_dict[g_idx]

        n_obs_local = int(np.sum(local_obs_mask))
        if n_obs_local == 0:
            continue

        local_ens = result[:, local_indices]  # (M, window_size)
        y_local = local_obs_vals[local_obs_mask]  # (n_obs_local,)

        # --- Local EnAKF update ---
        x_mean = np.mean(local_ens, axis=0)  # (window_size,)
        A = local_ens - x_mean  # (M, window_size)

        obs_pos = np.where(local_obs_mask)[0]  # positions in local window
        A_obs = A[:, obs_pos]  # (M, n_obs_local)

        # C_obs = A_obs^T A_obs / (M-1)
        C_obs = (A_obs.T @ A_obs) / (M - 1)  # (n_obs_local, n_obs_local)
        S = C_obs + observation_variance * np.eye(n_obs_local)

        # Compute Z = A_obs @ S^{-1} @ A_obs^T  -> (M, M)
        try:
            # Solve S X = A_obs^T  => X = S^{-1} A_obs^T  (n_obs, M)
            X = np.linalg.solve(S, A_obs.T)  # (n_obs_local, M)
            Z = A_obs @ X  # (M, n_obs) @ (n_obs, M) = (M, M)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            Z = A_obs @ S_inv @ A_obs.T

        # Transformation: T = I - (1/(M-1)) * Z
        T = np.eye(M) - (1.0 / (M - 1)) * Z

        # Update anomalies for all local variables
        A_new = T @ A  # (M, M) @ (M, window_size) = (M, window_size)
        updated_local = x_mean + A_new

        result[:, local_indices] = updated_local

    # Safety
    result = np.clip(result, -100.0, 100.0)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)

    return result
