import numpy as np
import time

# EVOLVE-BLOCK-START

# Module-level state
_inflation_state = {}
_start_time = None


def _get_time():
    global _start_time
    if _start_time is None:        _start_time = time.time()
    return time.time() - _start_time


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter with:
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

    # Compute background covariance (M-1 denominator)
    P_b = (X_anom.T @ X_anom) / (M - 1)  # (N, N)

    # Observation noise covariance: diagonal with obs_var
    # R = obs_var * I (for L observations)

    # Localization: apply exponential localization to the covariance
    # Use periodic distance for Lorenz-96
    state_idx = np.arange(N)
    loc_radius = 5.0  # localization length scale

    # Precompute localization matrix for the full state
    # For each pair (i, j), compute periodic distance
    loc_matrix = np.ones((N, N))
    for i in range(N):
        for j in range(N):
            d = abs(i - j)
            d = min(d, N - d)
            loc_matrix[i, j] = np.exp(-d / loc_radius)

    # Localized background covariance
    P_b_loc = P_b * loc_matrix  # element-wise (Schur product)

    # Observation operator: H selects observed indices
    # H is (L, N) with 1s at observed positions
    # Innovation covariance: S = H P_b_loc H^T + R
    # S[j, k] = P_b_loc[obs_idx[j], obs_idx[k]] * loc_matrix[obs_idx[j], obs_idx[k]] + (obs_var if j==k else 0)
    # Actually, we already applied localization to P_b, so:
    # S = P_b_loc[obs_idx[:,None], obs_idx[None,:]] + obs_var * I

    P_obs = P_b_loc[np.ix_(obs_idx, obs_idx)]  # (L, L)
    S = P_obs + obs_var * np.eye(L)

    # Compute S inverse (or use Cholesky for stability)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)

    # Innovation: y_obs - H X_b
    # H X_b for each member: X_b[:, obs_idx]
    H_X_b = X_b[:, obs_idx]  # (M, L)
    innov = y_obs - H_X_b  # (M, L)

    # Kalman gain: K = P_b_loc H^T S^{-1}
    # P_b_loc H^T is (N, L): for each state i and obs j, P_b_loc[i, obs_idx[j]]
    P_Ht = P_b_loc[:, obs_idx]  # (N, L)
    K = P_Ht @ S_inv  # (N, L)

    # Analysis update: X_a = X_b + K (y_obs - H X_b)
    # For each member: X_a[:, i] = X_b[:, i] + sum_j K[i,j] * innov[:,j]
    X_a = X_b + innov @ K.T  # (M, N)

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
