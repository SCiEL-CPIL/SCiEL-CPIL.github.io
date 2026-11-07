"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    start = time.time()
    deadline = 290.0

    # Safety margins
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Normalize cost to have mean ~1.5 (since cost = 1 + mean(x), and x ~ U[0,1]^3)
    # This makes the cost-aware normalization more principled
    cost_mean = np.mean(cost_safe)
    cost_norm = cost_safe / cost_mean  # Now mean ~1

    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -15, 15)

    # EI in objective units
    EI = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)

    # KG-like approximation:
    # The value of information from evaluating x is related to how much it can
    # reduce the expected minimum. We approximate this with a combination of:
    # 1. EI (exploitation)
    # 2. A term that rewards points that are "far" from already-explored regions
    #    (approximated by high sigma, indicating uncertainty)
    # 3. Discounted by cost

    # Base: cost-aware EI
    base_score = EI / cost_norm

    # Exploration bonus: reward uncertainty, but with diminishing returns
    # Use sqrt(sigma) to avoid over-exploring very uncertain regions
    sigma_mean = np.mean(sigma_safe)
    sigma_norm = sigma_safe / sigma_mean
    exploration_bonus = np.sqrt(sigma_norm) * 0.3 / cost_norm

    # Combine
    scores = base_score + exploration_bonus

    # Check deadline before returning
    if time.time() - start > deadline:
        return np.where(np.isfinite(-mu / cost_norm), -mu / cost_norm, 0.0)

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
