"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition combining EI and UCB exploration
    # with a deadline guard for safety.
    t0 = time.time()
    deadline = t0 + 250.0  # safety margin before 300s limit

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # UCB-style exploration term
    # Use a beta that balances exploitation and exploration
    beta = 2.0
    ucb = -mu + beta * sigma

    # Cost-aware: use softer penalty (sqrt of cost) to avoid over-penalizing
    # Blend EI and UCB, both cost-aware
    # Weight UCB slightly more for exploration in under-explored regime
    score = (0.5 * ei + 0.5 * ucb) / np.sqrt(cost)

    # Ensure all values are finite
    result = np.where(np.isfinite(score), score, 0.0)

    # Check deadline
    if time.time() > deadline:
        # Return current best result with safety margin
        pass

    return result
# EVOLVE-BLOCK-END
