"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = 120.0  # wall-clock safety margin (evaluator kills at 150s)


def _finite_clip(x, lo=-100.0, hi=100.0):
    x = np.where(np.isfinite(x), x, 0.0)
    return np.clip(x, lo, hi)


def _enkf_batch_update(ensemble, obs_vec, obs_idx, obs_var, D, loc_width):
    """
    Batch EnKF update with Gaussian localization on periodic ring.
    """
    N = ensemble.shape[0]
    M = len(obs_idx)

    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean  # (N, D)

    # Observation-space anomalies
    obs_anoms = anomalies[:, obs_idx]  # (N, M)

    # Covariance in observation space
    P_obs = (obs_anoms.T @ obs_anoms) / (N - 1)  # (M, M)
    R = obs_var * np.eye(M)
    S = P_obs + R

    # Regularize S for numerical stability
    eigvals = np.linalg.eigvalsh(S)
    min_eig = np.min(eigvals)
    if min_eig < 1e-8:
        S += (1e-8 - min_eig) * np.eye(M)
    else:
        S += 1e-10 * np.eye(M)

    # Cross-covariance: P H^T = (1/(N-1)) A^T obs_anoms
    PHt = (anomalies.T @ obs_anoms) / (N - 1)  # (D, M)

    # Gaussian localization on periodic ring
    idx = np.arange(D)
    loc_mat = np.ones((D, M))
    for m_idx, obs_i in enumerate(obs_idx):
        dist = np.abs(idx - obs_i)
        dist = np.minimum(dist, D - dist)
        loc_mat[:, m_idx] = np.exp(-0.5 * (dist / loc_width) ** 2)

    PHt_loc = PHt * loc_mat

    # Gain: K = PHt_loc @ S^{-1}
    try:
        K = PHt_loc @ np.linalg.solve(S, np.eye(M))
    except np.linalg.LinAlgError:
        K = PHt_loc @ np.linalg.pinv(S)

    # Update: x_i += K^T (y - H x_i)
    innovations = obs_vec[None, :] - ensemble[:, obs_idx]  # (N, M)
    deltas = innovations @ K.T  # (N, D)
    updated = ensemble + deltas

    return updated


def analyze(ensemble, observations, observed_indices, observation_variance):
    X = np.asarray(ensemble, dtype=np.float64)
    y = np.asarray(observations, dtype=np.float64)
    obs = np.asarray(observed_indices, dtype=np.int64)
    R_var = float(observation_variance)

    n, p = X.shape
    if n < 2:
        return _finite_clip(X.copy())

    # Ensure finite input
    X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)
    X = np.clip(X, -100.0, 100.0)
    y = np.nan_to_num(y, nan=0.0, posinf=100.0, neginf=-100.0)
    y = y.flatten()

    # --- Two-pass batch EnKF with different localization widths ---
    # Pass 1: narrow localization for local accuracy
    X_pass1 = _enkf_batch_update(X, y, obs, R_var, p, loc_width=2.5)

    # Pass 2: wider localization to smooth spurious correlations
    X_pass2 = _enkf_batch_update(X_pass1, y, obs, R_var, p, loc_width=4.0)

    # --- Adaptive variance inflation ---
    m2 = X_pass2.mean(axis=0)
    An = X_pass2 - m2[None, :]

    if len(obs) > 1:
        obs_var_ens = np.var(An[:, obs], axis=0, ddof=1)
        # Expected observation variance from original ensemble + noise
        A_orig = X - X.mean(axis=0)[None, :]
        obs_anoms_orig = A_orig[:, obs]
        P_obs_orig = (obs_anoms_orig.T @ obs_anoms_orig) / (n - 1)
        obs_var_exp = np.diag(P_obs_orig) + R_var
        ratio = np.mean(obs_var_ens) / np.mean(obs_var_exp)
        alpha = np.clip(1.0 + 0.3 * (1.0 - ratio), 0.85, 1.4)
    else:
        alpha = 1.0
    X_pass2 = m2[None, :] + alpha * An

    # --- Small deterministic nudge on observed coords ---
    w = 0.08
    X_pass2[:, obs] += w * (y[None, :] - X_pass2[:, obs])

    out = _finite_clip(X_pass2)
    if (time.time() - _START) > _DEADLINE - 5.0:
        return out
    return out
# EVOLVE-BLOCK-END
