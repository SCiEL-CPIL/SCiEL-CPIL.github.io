"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: return a simple fallback if we're running out of time
    start = time.time()
    deadline = 290.0  # 40s buffer before 330s kill

    # Safety margins
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)

    EI = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)

    # Cost-aware: divide by cost
    scores = EI / cost_safe

    # Add a small exploration bonus to help escape local optima
    # This is inspired by inspiration 1, which added 0.1 * sigma / cost
    # But we keep it modest to not overwhelm the EI signal
    uncertainty_bonus = 0.05 * sigma_safe / cost_safe
    scores = scores + uncertainty_bonus

    # Check deadline before returning
    if time.time() - start > deadline:
        # Fallback: simple exploitation-based score
        return np.where(np.isfinite(-mu / cost_safe), -mu / cost_safe, 0.0)

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
