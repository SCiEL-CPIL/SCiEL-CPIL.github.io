"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_start_time = None
_inflation_state = {}


def _get_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Localized Ensemble Kalman Filter with:
    - Sequential per-observation block updates (inspiration 1 structure)
    - Exponential localization within blocks
    - Adaptive multiplicative inflation with temporal memory
    - Deadline guard
    """
    M, N = ensemble.shape
    obs_var = float(observation_variance)
    obs_idx = np.asarray(observed_indices, dtype=int)
    L = len(obs_idx)

    # Deadline guard
    elapsed = _get_time()
    if elapsed > 140.0:
        result = np.array(ensemble, dtype=np.float64).copy()
        obs_arr = np.asarray(observations, dtype=np.float64)
        if obs_arr.ndim == 1:
            obs_arr = np.tile(obs_arr, (M, 1))
        result[:, obs_idx] += 0.5 * (obs_arr - result[:, obs_idx])
        np.clip(result, -100.0, 100.0, out=result)
        return result

    X_b = np.array(ensemble, dtype=np.float64)
    y_obs = np.asarray(observations, dtype=np.float64)
    if y_obs.ndim == 1:
        y_obs = np.tile(y_obs, (M, 1))

    # Work ensemble that gets updated sequentially per observation
    X_work = X_b.copy()

    # Precompute periodic distances from each state variable to each observation
    state_idx = np.arange(N)
    # distances: (N, L)
    diffs = np.abs(state_idx[:, None] - obs_idx[None, :])
    distances = np.minimum(diffs, N - diffs)

    # Localization length scale (exponential)
    L_loc = 3.5

    # Block radius: include variables within this periodic distance
    R_loc = 5

    for j in range(L):
        oj = obs_idx[j]
        # Local block: variables within R_loc of observation j
        mask = distances[:, j] <= R_loc
        block = np.where(mask)[0]
        nblock = len(block)
        if nblock == 0:
            continue

        # Local ensemble
        X_local = X_work[:, block]  # (M, nblock)
        mean_local = np.mean(X_local, axis=0)
        X_local_anom = X_local - mean_local  # (M, nblock)

        # Find position of observation within block
        obs_pos = np.where(block == oj)[0][0]

        # Local covariance
        P_local = (X_local_anom.T @ X_local_anom) / (M - 1)  # (nblock, nblock)

        # Innovation variance
        S = P_local[obs_pos, obs_pos] + obs_var
        if S <= 1e-12:
            continue

        # Member innovations for this observation
        innov = y_obs[:, j] - X_local[:, obs_pos]  # (M,)

        # Localized cross-covariance: P_local[:, obs_pos] * exp(-dist/L_loc)
        local_dists = distances[block, j]  # (nblock,)
        loc_weights = np.exp(-local_dists / L_loc)  # (nblock,)
        P_local_obs = P_local[:, obs_pos] * loc_weights  # (nblock,)

        # Kalman gain for this block
        K_local = P_local_obs / S  # (nblock,)

        # Analysis update
        X_local_a = X_local + np.outer(innov, K_local)  # (M, nblock)

        # Spread preservation within block: inflate if spread decreased too much
        spread_b = np.std(X_local, axis=0, ddof=1)
        spread_a = np.std(X_local_a, axis=0, ddof=1)
        for i in range(nblock):
            if spread_b[i] > 1e-6 and spread_a[i] < spread_b[i] * 0.85:
                ratio = spread_b[i] / spread_a[i]
                ratio = min(ratio, 1.12)
                X_local_a[:, i] = mean_local[i] + ratio * (X_local_a[:, i] - mean_local[i])

        X_work[:, block] = X_local_a

    X_a = X_work

    # Global adaptive multiplicative inflation with temporal memory
    spread_b = np.std(X_b, axis=0, ddof=1)
    spread_a = np.std(X_a, axis=0, ddof=1)

    if N not in _inflation_state:
        _inflation_state[N] = np.ones(N)
    prev_inflation = _inflation_state[N]

    min_ratio = 0.82
    inflation_factors = np.ones(N)
    for i in range(N):
        if spread_b[i] > 1e-6:
            ratio = spread_a[i] / spread_b[i]
            if ratio < min_ratio:
                target = min(1.25, min_ratio / ratio)
                inflation_factors[i] = 0.5 * prev_inflation[i] + 0.5 * target
            else:
                inflation_factors[i] = 0.7 * prev_inflation[i] + 0.3 * 1.0
        else:
            inflation_factors[i] = 0.7 * prev_inflation[i] + 0.3 * 1.0

    _inflation_state[N] = inflation_factors.copy()

    if np.any(inflation_factors > 1.0):
        mean_a = np.mean(X_a, axis=0)
        X_a = mean_a[None, :] + inflation_factors[None, :] * (X_a - mean_a[None, :])

    # Safety
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    np.clip(X_a, -100.0, 100.0, out=X_a)

    if not np.all(np.isfinite(X_a)):
        X_a = np.array(ensemble, dtype=np.float64).copy()
        obs_arr = np.asarray(observations, dtype=np.float64)
        if obs_arr.ndim == 1:
            obs_arr = np.tile(obs_arr, (M, 1))
        X_a[:, obs_idx] += 0.5 * (obs_arr - X_a[:, obs_idx])
        np.clip(X_a, -100.0, 100.0, out=X_a)

    return X_a

# EVOLVE-BLOCK-END
