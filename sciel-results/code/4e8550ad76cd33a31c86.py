"""Cost-aware acquisition for Styblinski-Tang 2D BO."""
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """Return 2000 finite scores (higher = evaluate next)."""
    # Time guard: if we're close to deadline, return a safe fallback
    if time.time() > _DEADLINE - 2.0:
        # Fallback: prefer cheap points with some randomness via mu
        return -cost - 0.1 * mu

    n = mu.shape[0]

    # Clamp sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-4)

    # Expected Improvement (standard)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-aware: EI per unit cost
    # cost is in [1, 2]; use it directly
    ei_cost = ei / cost

    # UCB term with budget-aware exploration
    # Estimate remaining budget: we don't have total cost, but we can use
    # the average cost of candidates as a proxy for current evaluation cost.
    # A simpler structural choice: use a fixed exploration weight that
    # balances exploitation and exploration, but weight it by inverse cost
    # so cheap points get more exploration.
    beta = 0.5  # exploration weight
    ucb = mu + beta * sigma
    # For minimization, lower is better; UCB for minimization is mu - beta*sigma
    # But we want to explore where sigma is high. Use:
    ucb_min = mu - beta * sigma  # lower = more uncertain / potentially better
    # Normalize ucb_min to be comparable to ei_cost
    # Use a scaled version
    ucb_term = -ucb_min  # higher = more exploration potential

    # Combine: weighted sum of cost-normalized EI and exploration
    # Use a dynamic weight: more exploration when EI is small
    ei_max = np.max(ei_cost) if np.max(ei_cost) > 0 else 1.0
    ei_norm = ei_cost / (ei_max + 1e-9)
    ucb_max = np.max(ucb_term) if np.max(ucb_term) > 0 else 1.0
    ucb_norm = ucb_term / (ucb_max + 1e-9)

    # Structural change: use a product-like combination that favors
    # points with both high EI and reasonable cost, plus exploration
    score = ei_norm + 0.3 * ucb_norm

    # Additional structural bias: prefer cheaper points slightly
    # cost is in [1,2], so (2 - cost) is in [0,1], higher = cheaper
    cost_bias = (2.0 - cost) * 0.1
    score = score + cost_bias

    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
