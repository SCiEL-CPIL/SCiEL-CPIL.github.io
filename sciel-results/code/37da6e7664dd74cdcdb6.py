"""Cost-aware acquisition for levy 2d BO loop."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_deadline = time.time() + 300.0
_MARGIN = 2.0
_EPS = 1e-9


def acquisition(mu, sigma, f_best, cost):
    """Return 2000 finite scores (higher = evaluate next).

    Uses a blend of cost-normalized Expected Improvement (EI),
    Upper Confidence Level (UCL) for exploration, and a small
    diversity bonus to avoid premature convergence.
    """
    global _deadline
    if time.time() > _deadline - _MARGIN:
        # Safety: return a valid finite array quickly.
        return np.full(2000, 0.0)

    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()

    n = mu.size
    if n != 2000:
        # Pad or truncate to expected size, then clip to finite.
        if n < 2000:
            mu = np.pad(mu, (0, 2000 - n), constant_values=mu[-1] if n else 0.0)
            sigma = np.pad(sigma, (0, 2000 - n), constant_values=sigma[-1] if n else 1.0)
            cost = np.pad(cost, (0, 2000 - n), constant_values=cost[-1] if n else 1.0)
        else:
            mu = mu[:2000]
            sigma = sigma[:2000]
            cost = cost[:2000]

    # Guard against non-finite inputs.
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-3)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Expected Improvement per unit cost.
    improvement = f_best - mu
    z = improvement / (sigma + _EPS)
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    ei_cost = ei / cost

    # UCL: balance exploitation (mu) and exploration (sigma),
    # scaled by cost to stay comparable with EI.
    # beta grows modestly to encourage exploration in uncertain regions.
    beta = 0.5
    ucl = -mu + beta * sigma
    ucl_cost = ucl / cost

    # Diversity bonus: prefer candidates far from the current best
    # (lowest mu) to avoid clustering. Use rank of mu as a proxy.
    # Lower mu = better, so higher rank (worse) gets a small bonus.
    mu_rank = np.argsort(np.argsort(mu)) / max(n - 1, 1)  # 0=best, 1=worst
    diversity = 0.05 * mu_rank / cost

    # Blend: EI is the primary driver, UCL adds exploration,
    # diversity prevents local trapping.
    scores = 0.7 * ei_cost + 0.25 * ucl_cost + diversity

    # Ensure finiteness and reasonable scale.
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.clip(scores, -1e6, 1e6)

    return scores
# EVOLVE-BLOCK-END
