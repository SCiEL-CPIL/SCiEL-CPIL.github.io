# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) with adaptive decay.
    For sphere 1D minimization (x^2 on [-5,5]), uses:
      score = -(mu - beta * sigma)
    where beta decays as f_best approaches 0, shifting from exploration to exploitation.
    """
    start_time = time.time()
    deadline = 300.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Sanitize
    sigma_safe = np.maximum(sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-12)

    # Adaptive beta: starts high for exploration, decays as we find better solutions
    # f_best is in [0, 25] for sphere 1D
    # When f_best is large (early), beta ~ 2.0 (heavy exploration)
    # When f_best ~ 0 (converged), beta ~ 0.1 (pure exploitation)
    beta = 0.1 + 1.9 * np.exp(-f_best / 5.0)

    # UCB score for minimization: we want low mu and high sigma
    # score = -(mu - beta * sigma) = -mu + beta * sigma
    # Higher score = better candidate to evaluate next
    score = -mu + beta * sigma_safe

    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)

    # Deadline guard
    if time.time() - start_time > deadline:
        return np.zeros_like(score)

    return score
# EVOLVE-BLOCK-END
