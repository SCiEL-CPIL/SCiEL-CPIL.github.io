# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    t0 = time.time()

    # Clamp sigma to a numerically safe floor
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)

    # Standardised distance of each candidate from the current best
    z = (f_best - mu) / sigma_safe

    # Expected Improvement for minimisation
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # UCB-style exploration: balances exploitation (mu) against uncertainty (sigma)
    # Beta controls the exploration strength; 1.5 is a common default for GP-UCB
    beta = 1.5
    ucb = -mu + beta * sigma_safe

    # Blend EI and UCB: EI focuses on improving the best, UCB explores high-uncertainty regions
    # Weighting: 0.6 EI + 0.4 UCB gives a balance between exploitation and exploration
    score = 0.6 * ei + 0.4 * ucb

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    # Deadline guard: if we are close to the 330 s wall-clock limit, return immediately
    if time.time() - t0 > 310.0:
        return score

    return score
# EVOLVE-BLOCK-END
