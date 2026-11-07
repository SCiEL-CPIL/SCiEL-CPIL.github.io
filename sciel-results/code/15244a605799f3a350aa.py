"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_T0 = time.time()
_DEADLINE = 320.0


def _time_ok():
    return (time.time() - _T0) < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    if not _time_ok():
        # Fallback: return finite scores based on low mu (exploitation)
        score = -mu / (1.0 + np.abs(mu))
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        return score

    # Ensure positive sigma
    sigma_safe = np.maximum(sigma, 1e-10)

    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    delta = f_best - mu
    z = delta / sigma_safe

    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10.0, 10.0)

    ei = delta * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)

    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)

    # Cost-aware adjustment:
    # Cost is in [1, 2] (since 1 + mean(x_scaled) where x_scaled in [0,1])
    # We want to discount high-cost points but not too aggressively
    # Use a moderate cost discount: score = EI * (1 / cost^alpha)
    # alpha=1 gives linear discount, alpha=0.5 gives sqrt discount
    # Try alpha=0.7 for moderate cost awareness
    cost_discount = 1.0 / (cost ** 0.7)

    score = ei * cost_discount

    # Add a small exploration bonus for high uncertainty regions
    # This helps escape local minima in multimodal landscapes
    # Scale exploration bonus to be comparable to EI values
    # Typical EI values might be in range [0, 10] for this problem
    exploration_bonus = 0.1 * sigma_safe

    score = score + exploration_bonus

    # Ensure finiteness
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    return score
# EVOLVE-BLOCK-END
