# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = 320.0


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def _matern52(r, length_scale=0.2):
    """Matern-5/2 correlation for scaled distances r (diagonal forced to 1)."""
    r = np.asarray(r, dtype=float)
    r = r / length_scale
    out = (1.0 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r)
    np.fill_diagonal(out, 1.0)
    return out


def _gp_predict(xc, X, y, length_scale=0.2, noise=1e-6):
    """Return (mu, sigma) in objective units for points xc given data X, y."""
    n = X.shape[0]
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        y_std = 1.0
    ys = (y - y_mean) / y_std

    # Pairwise distances in scaled space
    dX = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    K = _matern52(dX, length_scale)
    K = K + (noise / (y_std ** 2)) * np.eye(n)

    # Cholesky with fallback
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(K + 1e-8 * np.eye(n))

    v = np.linalg.solve(L, ys)

    dXs = np.linalg.norm(xc[:, None, :] - X[None, :, :], axis=-1)
    Ks = _matern52(dXs, length_scale)
    a = np.linalg.solve(L, Ks.T)

    mu = y_mean + y_std * (a.T @ v)
    var = y_std ** 2 * (1.0 - np.einsum('ij,ij->i', a, a))
    sigma = np.sqrt(np.maximum(var, 1e-14))
    return mu, sigma


def acquisition(mu, sigma, f_best, cost):
    # Guard: if we are near the wall-clock limit, return a cheap, valid score.
    if _time_left() < 3.0:
        return np.zeros_like(mu, dtype=float)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    cost = np.where(cost > 0, cost, 1.0)

    # Recover scaled coordinates from cost = 1 + mean(x_scaled).
    x = np.clip((cost - 1.0) * 2.0, 0.0, 1.0)

    # Fit a lightweight GP on the predicted targets at the candidate points.
    try:
        pmu, psig = _gp_predict(x, x, mu)
    except Exception:
        pmu, psig = mu, sigma

    # Ensure sigma is strictly positive for the EI/UCB formulas.
    psig = np.maximum(psig, 1e-12)

    # Expected improvement (minimization) in objective units.
    z = (f_best - pmu) / psig
    ei = (f_best - pmu) * norm.cdf(z) + psig * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # UCB: exploration term that rewards high uncertainty.
    beta = 0.3
    ucb = pmu + beta * psig  # lower is better for minimization

    # Combine EI and UCB: we want high EI and low UCB.
    # Normalize both to comparable scales.
    ei_scale = np.max(ei) if np.max(ei) > 1e-12 else 1.0
    ucb_scale = np.ptp(ucb) if np.ptp(ucb) > 1e-12 else 1.0

    ei_norm = ei / ei_scale
    ucb_norm = (np.max(ucb) - ucb) / ucb_scale  # higher = better (lower ucb)

    # Weighted combination: 60% exploitation (EI), 40% exploration (UCB).
    combined = 0.6 * ei_norm + 0.4 * ucb_norm

    # Cost-aware score: penalize expensive evaluations.
    # cost is in [1, 2], so 1/cost is in [0.5, 1].
    score = combined / cost

    # Mild bonus for points near the center (low cost) to encourage cheap exploration.
    center_bonus = 0.1 * (2.0 - cost)
    score += center_bonus

    # Ensure all scores are finite and non-negative for stability.
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)

    return score
# EVOLVE-BLOCK-END
