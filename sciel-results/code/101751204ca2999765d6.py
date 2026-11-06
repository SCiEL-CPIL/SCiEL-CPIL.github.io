"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = 120.0  # wall-clock safety margin (evaluator kills at 150s)


def _finite_clip(x, lo=-100.0, hi=100.0):
    x = np.where(np.isfinite(x), x, 0.0)
    return np.clip(x, lo, hi)


def _gaspari_cohn(d, r0, r1):
    """Gaspari-Cohn correlation function for periodic distance d."""
    loc = np.ones_like(d)
    mask = d > r0
    if not np.any(mask):
        return loc
    x = d[mask] / (r1 - r0)
    loc[mask] = 0.25 * x**4 * (5.0 * x - 9.0) + 0.5 * x**2
    return np.clip(loc, 0.0, 1.0)


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

    # --- Localization: Gaspari-Cohn taper on the periodic ring ---
    idx = np.arange(p)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, p - dist)         # periodic distance
    r0, r1 = 5.0, 10.0
    loc = _gaspari_cohn(dist, r0, r1)

    # --- Compute localized sample covariance ---
    # P = (A.T @ A) / (n-1), then apply localization
    P = (A.T @ A) / (n - 1)                   # (p, p)
    P = P * loc

    # --- EnSRF: sequential square-root update per observed coordinate ---
    X_work = X.copy()
    k = len(obs)
    R_std = np.sqrt(R_var)

    for j in range(k):
        idx_obs = obs[j]
        
        # Current ensemble mean and anomalies
        m_cur = X_work.mean(axis=0)
        A_cur = X_work - m_cur[None, :]
        
        # Compute correlation between state variable j and all variables
        # Using localized covariance structure
        # P[j, :] gives covariance of variable j with all others
        P_j = P[j, :]  # (p,)
        
        # Compute gain vector for this observation
        # K_j = P[j, :] / (P[j,j] + R_var)
        denom = P[j, j] + R_var
        if denom < 1e-12:
            continue
        K_j = P_j / denom  # (p,)
        
        # Innovation for this observation
        innov = y[j] - X_work[:, idx_obs]  # (n,)
        
        # Update each member: x_i += K_j * (y_j - x_i,j)
        # This is the standard EnKF update, but we do it sequentially
        # to maintain consistency (EnSRF-like behavior)
        X_work += np.outer(innov, K_j)
        
        # Check deadline
        if (time.time() - _START) > _DEADLINE - 5.0:
            return _finite_clip(X_work)

    # --- Adaptive variance inflation ---
    m2 = X_work.mean(axis=0)
    An = X_work - m2[None, :]
    
    # Compare observed-dimension ensemble spread to expected innovation std
    if k > 1:
        obs_var_ens = np.var(An[:, obs], axis=0, ddof=1)
        # Expected innovation variance: P_obs + R_var
        P_obs_diag = np.diag(P[np.ix_(obs, obs)])
        obs_var_exp = P_obs_diag + R_var
        ratio = np.mean(obs_var_ens) / np.mean(obs_var_exp)
        # Inflate if spread is too small
        alpha = np.clip(1.0 + 0.4 * (1.0 - ratio), 0.9, 1.5)
    else:
        alpha = 1.0
    X_work = m2[None, :] + alpha * An

    # --- Small deterministic nudge on observed coords to ensure consistency ---
    w = 0.05
    X_work[:, obs] += w * (y[None, :] - X_work[:, obs])

    out = _finite_clip(X_work)
    return out
# EVOLVE-BLOCK-END
