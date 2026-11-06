"""Combined ensemble assimilation for Lorenz-96 partial observation using Deterministic EnSRF with Gaspari-Cohn localization."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_start_time = None

def _get_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _gaspari_cohn(d, r):
    """
    Gaspari-Cohn correlation function.
    d: distance
    r: correlation length (half-width)
    """
    d = np.abs(d)
    # Ensure d/r is within [0, 2] for the function definition
    dr = d / r
    # For d > 2r, correlation is 0
    mask = dr <= 2.0
    result = np.zeros_like(dr)
    
    # Piecewise polynomial
    # For 0 <= dr <= 1
    m1 = mask & (dr <= 1.0)
    a1 = -2.0 / 15.0 * dr[m1] ** 5 + 8.0 / 3.0 * dr[m1] ** 4 - 5.0 * dr[m1] ** 3 + 4.0 - 4.0 / 3.0 * dr[m1]
    result[m1] = a1
    
    # For 1 < dr <= 2
    m2 = mask & (dr > 1.0)
    a2 = 4.0 / 3.0 * dr[m2] ** 5 - 5.0 * dr[m2] ** 4 + 10.0 * dr[m2] ** 3 - 5.0 * dr[m2] ** 2 + 1.0
    result[m2] = a2 / 3.0
    
    return result

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Deterministic Ensemble Square Root Filter (EnSRF) with:
    - Gaspari-Cohn localization
    - Adaptive multiplicative inflation
    - Deadline guard
    """
    M, N = ensemble.shape
    obs_var = float(observation_variance)
    obs_idx = np.asarray(observed_indices, dtype=int)

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

    L = len(obs_idx)

    # Background statistics
    mean_b = np.mean(X_b, axis=0)
    W_b = X_b - mean_b  # (M, N)

    # Background covariance (N x N)
    P_b = (W_b.T @ W_b) / (M - 1)

    # Observation operator components
    P_oo = P_b[np.ix_(obs_idx, obs_idx)]
    P_xo = P_b[:, obs_idx]  # (N, L)

    # Innovation covariance
    S = P_oo + obs_var * np.eye(L)

    # Gaspari-Cohn localization of cross-covariance
    state_idx = np.arange(N)
    diffs = np.abs(state_idx[:, None] - obs_idx[None, :])
    distances = np.minimum(diffs, N - diffs)  # (N, L)

    L_loc = 3.0  # correlation length scale (tuned)
    loc_matrix = _gaspari_cohn(distances, L_loc)  # (N, L)

    P_xo_loc = P_xo * loc_matrix  # (N, L)

    # Kalman gain: K = P_xo_loc @ S^{-1}
    try:
        # Solve S X = P_xo_loc.T -> X = S^{-1} P_xo_loc.T
        # K = P_xo_loc @ S^{-1}
        # We want K (N, L).
        # P_xo_loc (N, L). S (L, L).
        # K = P_xo_loc @ inv(S)
        # To avoid explicit inverse:
        # Solve S.T @ Y.T = P_xo_loc.T -> Y = P_xo_loc @ S^{-1}
        S_inv_Pxot = np.linalg.solve(S.T, P_xo_loc.T)  # (L, N)
        K = S_inv_Pxot.T  # (N, L)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        K = P_xo_loc @ S_inv  # (N, L)

    # Mean update
    d_mean = y_obs[0] - mean_b[obs_idx]  # innovation vector for mean (L,)
    mean_a = mean_b + K @ d_mean  # (N,)

    # Perturbation update (Deterministic EnSRF)
    # We want to update W_a such that:
    # 1. Mean is preserved (we re-center later)
    # 2. Covariance is updated as P_a = (I - K H) P_b (I - K H)^T + K R K^T
    #    But for EnSRF, we directly update perturbations.
    #    Standard EnSRF update for perturbations W_b (M, N) -> W_a (M, N):
    #    W_a = (I - K H) W_b
    #    This reduces variance. We need to inflate.
    
    H_W_b = W_b[:, obs_idx]  # (M, L)
    # W_a = W_b - K @ H_W_b
    # W_b is (M, N). K is (N, L). H_W_b is (M, L).
    # K @ H_W_b is (N, M).
    # We need (M, N) result.
    # W_a = W_b - (K @ H_W_b).T
    K_HW_b_T = (K @ H_W_b.T).T  # (M, N)
    W_a = W_b - K_HW_b_T  # (M, N)

    # Adaptive multiplicative inflation
    # Estimate innovation variance from ensemble
    # Theoretical innovation variance: diag(S)
    # Sample innovation variance: var(y_obs - H X_b)
    innovations = y_obs - X_b[:, obs_idx]  # (M, L)
    sample_innov_var = np.var(innovations, axis=0)  # (L,)
    theo_innov_var = np.diag(S)  # (L,)
    
    # Ratio of sample to theoretical
    # If sample > theo, we need to inflate
    # We use the mean ratio
    ratio = np.mean(sample_innov_var / (theo_innov_var + 1e-10))
    
    # Clip ratio to avoid extreme inflation
    ratio = np.clip(ratio, 1.0, 1.2)
    
    # Inflation factor is sqrt(ratio) for standard deviation, but we apply to W_a
    # W_a is the perturbation. Var(W_a) ~ 1/ratio^2? No.
    # We want Var(W_a_inflated) = Var(W_a) * ratio
    # So W_a_inflated = W_a * sqrt(ratio)
    inflation_factor = np.sqrt(ratio)
    
    # Apply inflation
    W_a *= inflation_factor

    # Re-center perturbations to ensure mean is exactly mean_a
    mean_W_a = np.mean(W_a, axis=0)
    W_a -= mean_W_a

    # Reconstruct ensemble
    X_a = mean_a[None, :] + W_a  # (M, N)

    # Safety: ensure finite and clip
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    np.clip(X_a, -100.0, 100.0, out=X_a)

    return X_a

# EVOLVE-BLOCK-END
