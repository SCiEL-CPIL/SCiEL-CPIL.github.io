"""Rewritten: iterated ensemble Kalman smoother with adaptive localization."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = 118.0  # wall-clock safety margin (evaluator kills at 150s)


def _finite_clip(x, lo=-100.0, hi=100.0):
    x = np.where(np.isfinite(x), x, 0.0)
    return np.clip(x, lo, hi)


def _gaspari_cohn(dist, r0, r1):
    """Periodic Gaspari-Cohn taper. Returns (p,p) array in [0,1]."""
    p = dist.shape[0]
    loc = np.ones((p, p))
    mask = dist > r0
    if np.any(mask):
        d = dist[mask]
        x = np.clip(d / (r1 - r0), 0.0, 1.0)
        val = 0.25 * x**4 * (5.0 * x - 9.0) + 0.5 * x**2
        loc[mask] = val
    return np.clip(loc, 0.0, 1.0)


def analyze(ensemble, observations, observed_indices, observation_variance):
    X = np.asarray(ensemble, dtype=np.float64).copy()
    y = np.asarray(observations, dtype=np.float64)
    obs = np.asarray(observed_indices, dtype=np.int64)
    R_var = float(observation_variance)

    n, p = X.shape
    if n < 2:
        return _finite_clip(X)

    k = len(obs)
    if k == 0:
        return _finite_clip(X)

    # --- Precompute periodic distance matrix (cheap, p=20) ---
    idx = np.arange(p)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, p - dist)

    # --- Iterated ensemble Kalman smoother ---
    # Multiple passes: each pass re-estimates mean/cov from the current
    # ensemble, then applies a Kalman update. This is a first-order
    # approximation to the full nonlinear smoothing / MLE update.
    n_iter = 3
    for _ in range(n_iter):
        if (time.time() - _START) > _DEADLINE:
            break

        m = X.mean(axis=0)
        A = X - m[None, :]
        P = (A.T @ A) / (n - 1)

        # --- Adaptive localization width from innovation statistics ---
        # If innovations are large relative to expected, the prior is
        # overconfident -> widen localization (less aggressive taper).
        # If innovations are small, the prior is well-calibrated ->
        # tighter localization is safe.
        innov0 = y[None, :] - X[:, obs]
        innov_var = np.var(innov0, axis=0, ddof=1) if k > 1 else np.array([np.var(innov0[:, 0])])
        Po_diag = np.diag(P[np.ix_(obs, obs)])
        expected_innov_var = Po_diag + R_var
        # ratio > 1 means ensemble spread too small (overconfident)
        ratio = float(np.mean(innov_var)) / float(np.mean(expected_innov_var))
        # Map ratio to localization radii:
        # ratio ~ 1.0 -> r0=4, r1=9 (baseline)
        # ratio > 1  -> wider (less localization)
        # ratio < 1  -> tighter (more localization)
        scale = np.clip(ratio, 0.5, 3.0)
        r0 = 4.0 * np.sqrt(scale)
        r1 = 9.0 * np.sqrt(scale)
        r0 = float(np.clip(r0, 2.0, 10.0))
        r1 = float(np.clip(r1, 5.0, 14.0))

        loc = _gaspari_cohn(dist, r0, r1)
        P_loc = P * loc

        # --- Kalman gain via stable solve ---
        Po = P_loc[np.ix_(obs, obs)]
        Ho = P_loc[obs, :].T  # (p, k)
        S = Po + R_var * np.eye(k)
        try:
            K = Ho @ np.linalg.solve(S, np.eye(k))
        except np.linalg.LinAlgError:
            K = Ho @ np.linalg.pinv(S)

        # --- Innovation and update ---
        innov = y[None, :] - X[:, obs]
        X = X + innov @ K.T

        # --- Statistical variance inflation ---
        # After update, check if the ensemble spread in observed space
        # matches the expected spread. Inflate/deflate to maintain
        # consistency. Use a damped correction to avoid oscillation.
        m2 = X.mean(axis=0)
        An = X - m2[None, :]
        if k > 1:
            obs_var_ens = np.var(An[:, obs], axis=0, ddof=1)
            # Expected post-update spread: for linear Gaussian,
            # post-update variance of obs = R_var * (1 - K diag effect)
            # Simpler: compare to R_var as a floor
            expected_post = np.maximum(np.diag(Po) * 0.5, R_var * 0.5)
            ratio_post = float(np.mean(obs_var_ens)) / float(np.mean(expected_post))
            # Damped inflation: alpha in [0.9, 1.3]
            alpha = np.clip(1.0 + 0.2 * (1.0 - ratio_post), 0.9, 1.3)
        else:
            alpha = 1.0
        X = m2[None, :] + alpha * An

    # --- Final safety: ensure observed coords are reasonably close ---
    # Small deterministic nudge on observed coords to guarantee
    # the analysis mean tracks observations (guards against
    # filter divergence in chaotic systems).
    w = 0.05
    X[:, obs] += w * (y[None, :] - X[:, obs])

    return _finite_clip(X)
# EVOLVE-BLOCK-END
