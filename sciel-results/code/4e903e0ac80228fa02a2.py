import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 140.0

def analyze(ensemble, observations, observed_indices, observation_variance):
    if time.time() > _DEADLINE:
        return np.array(ensemble, dtype=np.float64)

    X_b = np.array(ensemble, dtype=np.float64)
    y = np.array(observations, dtype=np.float64)
    obs_idx = np.array(observed_indices, dtype=int)
    R_val = float(observation_variance)

    M, N = X_b.shape
    N_obs = len(obs_idx)

    # --- Block-local sequential EnSRF with LET ---
    # Process observations in blocks of 4 to reduce spurious long-range correlations
    block_size = 4
    n_blocks = (N_obs + block_size - 1) // block_size

    # Adaptive inflation based on global innovation statistics
    mean_b = np.mean(X_b, axis=0)
    W_b = X_b - mean_b
    P_b = (W_b.T @ W_b) / (M - 1)

    # Global innovation for inflation
    d_global = y - mean_b[obs_idx]
    P_oo_global = P_b[np.ix_(obs_idx, obs_idx)]
    expected_var = np.mean(np.diag(P_oo_global + np.eye(N_obs) * R_val))
    actual_var = np.var(d_global) + 1e-12
    ratio = actual_var / (expected_var + 1e-12)
    inflation = np.sqrt(np.clip(ratio, 0.9, 1.5))
    inflation = np.clip(inflation, 1.0, 1.06)

    X_work = X_b.copy()
    W_work = X_work - np.mean(X_work, axis=0)

    for b in range(n_blocks):
        i_start = b * block_size
        i_end = min((b + 1) * block_size, N_obs)
        idx_block = obs_idx[i_start:i_end]
        y_block = y[i_start:i_end]
        nb = len(idx_block)

        # Recompute mean and spread from current ensemble state
        mean_cur = np.mean(X_work, axis=0)
        W_cur = X_work - mean_cur
        P_cur = (W_cur.T @ W_cur) / (M - 1)

        # Block covariances
        P_oo = P_cur[np.ix_(idx_block, idx_block)]
        P_xo = P_cur[:, idx_block]

        # LET localization: Gaussian decay centered on each observed variable
        dist = np.abs(np.arange(N)[:, np.newaxis] - idx_block[np.newaxis, :])
        dist = np.minimum(dist, N - dist)
        L = 2.0
        loc = np.exp(-0.5 * (dist / L) ** 2)

        P_xo_loc = P_xo * loc

        # Innovation
        d = y_block - mean_cur[idx_block]

        # Innovation covariance
        S = P_oo + np.eye(nb) * R_val
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        K = P_xo_loc @ S_inv

        # Update mean
        mean_a = mean_cur + K @ d

        # EnSRF deterministic update on perturbations
        H_W_T = W_cur.T[idx_block, :]
        K_HW = K @ H_W_T
        W_a_T = W_cur.T - K_HW

        # LET spread correction: preserve local spread
        # For each variable in the block, preserve the original spread weighted by localization
        for j in range(nb):
            loc_j = loc[:, j]
            var_orig = np.var(W_cur.T[:, loc_j > 0.05], axis=1) if np.any(loc_j > 0.05) else 0.0
            var_new = np.var(W_a_T[:, loc_j > 0.05], axis=1) if np.any(loc_j > 0.05) else 0.0
            mask = loc_j > 0.05
            if np.any(mask) and np.any(var_new > 1e-12):
                scale = np.sqrt(np.maximum(var_orig[mask], 1e-8) / np.maximum(var_new[mask], 1e-8))
                scale = np.clip(scale, 0.8, 1.2)
                W_a_T[mask, j] *= scale

        # Re-center
        mean_W = np.mean(W_a_T, axis=1)
        W_a_T -= mean_W[:, np.newaxis]

        X_work = mean_a + W_a_T.T

    # Final adaptive inflation
    X_work = np.mean(X_work, axis=0) + (X_work - np.mean(X_work, axis=0)) * inflation

    # Safety
    X_work = np.clip(X_work, -100.0, 100.0)
    X_work = np.nan_to_num(X_work, nan=0.0, posinf=100.0, neginf=-100.0)

    return X_work
