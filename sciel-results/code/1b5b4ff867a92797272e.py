"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = 120.0  # wall-clock safety margin (evaluator kills at 150s)


def _finite_clip(x, lo=-100.0, hi=100.0):
    x = np.where(np.isfinite(x), x, 0.0)
    return np.clip(x, lo, hi)


def analyze(ensemble, observations, observed_indices, observation_variance):
    X = np.asarray(ensemble, dtype=np.float64)
    y = np.asarray(observations, dtype=np.float64)
    obs = np.asarray(observed_indices, dtype=np.int64)
    R_var = float(observation_variance)

    n, p = X.shape
    if n < 2:
        return _finite_clip(X.copy())

    # --- Prior mean and anomalies ---
    m = X.mean(axis=0)
    A = X - m[None, :]                       # (n, p) anomalies
    # Sample covariance (unbiased)
    P = (A.T @ A) / (n - 1)                   # (p, p)

    # --- Localization: Gaspari-Cohn taper on the periodic ring ---
    idx = np.arange(p)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, p - dist)         # periodic distance
    r0, r1 = 3.0, 7.0
    loc = np.ones((p, p))
    mask = dist > r0
    d = dist[mask]
    if np.any(mask):
        x = d / (r1 - r0)
        loc[mask] = 0.25 * x**4 * (5.0 * x - 9.0) + 0.5 * x**2
    loc = np.clip(loc, 0.0, 1.0)
    P = P * loc

    # --- Observation operator: identity on obs indices ---
    k = len(obs)
    Po = P[np.ix_(obs, obs)]                  # (k, k)
    Ho = P[obs, :].T                          # (p, k)   = P[:, obs]
    # Stable solve instead of explicit inverse
    try:
        K = Ho @ np.linalg.solve(Po + R_var * np.eye(k), np.eye(k))
    except np.linalg.LinAlgError:
        K = Ho @ np.linalg.pinv(Po + R_var * np.eye(k))

    # --- Per-member innovation and update ---
    innov = y[None, :] - X[:, obs]            # (n, k)
    X_new = X + innov @ K.T                   # (n, p)

    # --- Adaptive variance inflation (keep spread healthy against overconfidence) ---
    m2 = X_new.mean(axis=0)
    An = X_new - m2[None, :]
    # Compare observed-dimension ensemble spread to expected innovation std
    if k > 1:
        obs_var_ens = np.var(An[:, obs], axis=0, ddof=1)
        obs_var_exp = np.diag(Po + R_var * np.eye(k))
        ratio = np.mean(obs_var_ens) / np.mean(obs_var_exp)
        alpha = np.clip(1.0 + 0.4 * (1.0 - ratio), 0.80, 1.5)
    else:
        alpha = 1.0
    X_new = m2[None, :] + alpha * An

    # --- Small deterministic nudge on observed coords ---
    w = 0.05
    X_new[:, obs] += w * (y[None, :] - X_new[:, obs])

    out = _finite_clip(X_new)
    if (time.time() - _START) > _DEADLINE - 5.0:
        return out
    return out
# EVOLVE-BLOCK-END
