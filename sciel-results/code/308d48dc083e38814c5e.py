import numpy as np
import time

# EVOLVE-BLOCK-START

# Module-level state
_inflation_state = {}
_start_time = None


def _get_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter with:
    - Full background covariance
    - Exponential localization (periodic distance)
    - Adaptive multiplicative inflation with temporal memory
    - Deadline guard for safety
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

    mean_b = np.mean(X_b, axis=0)
    X_anom = X_b - mean_b  # (M, N)

    # Background covariance (N, N)
    P_b = (X_anom.T @ X_anom) / (M - 1)

    # Exponential localization with periodic distance
    # Correlation decay length: 4 (tunable)
    decay_length = 4.0

    # Compute periodic distances between all pairs
    state_idx = np.arange(N)
    dists = np.abs(state_idx[:, None] - state_idx[None, :])
    dists = np.minimum(dists, N - dists)  # periodic

    # Localization weights: exp(-d^2 / (2 * decay_length^2))
    loc_weights = np.exp(-dists ** 2 / (2.0 * decay_length ** 2))

    # Localized background covariance
    P_loc = P_b * loc_weights

    # Observation indices
    obs_pos = obs_idx  # (L,)

    # Innovation variance matrix: S = P_loc[obs, obs] + I * obs_var
    S = P_loc[np.ix_(obs_pos, obs_pos)] + np.eye(L) * obs_var

    # Cholesky of S for stability
    try:
        L_S = np.linalg.cholesky(S)
        S_inv = np.linalg.solve(L_S.T, np.linalg.solve(L_S, np.eye(L)))
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)

    # Kalman gain: K = P_loc[:, obs] @ S_inv  (N, L)
    P_cross = P_loc[:, obs_pos]  # (N, L)
    K = P_cross @ S_inv  # (N, L)

    # Innovations: (M, L)
    innovations = y_obs - X_b[:, obs_pos]  # (M, L)

    # Analysis update: X_a = X_b + K @ innovations.T  ->  X_a[m, :] = X_b[m, :] + innovations[m, :] @ K.T
    X_a = X_b + innovations @ K.T  # (M, N)

    # Adaptive multiplicative inflation with temporal memory
    spread_b = np.std(X_b, axis=0, ddof=1)
    spread_a = np.std(X_a, axis=0, ddof=1)

    if N not in _inflation_state:
        _inflation_state[N] = np.ones(N)
    prev_inflation = _inflation_state[N]

    min_ratio = 0.85
    inflation_factors = np.ones(N)
    for i in range(N):
        if spread_b[i] > 1e-6:
            ratio = spread_a[i] / spread_b[i]
            if ratio < min_ratio:
                target = min(1.15, min_ratio / ratio)
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
