"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function for Bayesian optimization.
    
    Instead of computing a deterministic acquisition score (like EI/cost),
    we draw a single random sample from the GP posterior and select the
    candidate that minimizes the sampled objective divided by its cost.
    
    This treats the objective as a stochastic reward-to-go ratio, which
    provides a natural exploration-exploitation balance and inherently
    accounts for uncertainty in a way that differs from EI or PI.
    
    The Thompson Sampling approach:
    1. Draw z ~ N(0, 1) for each candidate
    2. Sample f_sample = mu + sigma * z (posterior sample)
    3. Score = -f_sample / cost (we want to minimize f, so negative score for minimization)
    4. Return scores where higher = evaluate next
    
    This is structurally different from EI/cost or PI/cost because:
    - It doesn't condition on f_best explicitly in the score computation
    - The randomness provides exploration that adapts to the posterior uncertainty
    - The cost division is applied to the sampled objective, not to an improvement metric
    """
    # Ensure numerical stability
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Draw a single random sample from the GP posterior
    # z ~ N(0, 1) for each of the 2000 candidates
    z = np.random.randn(len(mu))
    
    # Posterior sample: f_sample = mu + sigma * z
    # Since we're minimizing, lower f_sample is better
    f_sample = mu + sigma_safe * z
    
    # Cost-aware score: we want to minimize f_sample / cost
    # So higher score = lower f_sample / cost
    # score = -f_sample / cost (negate so that higher is better)
    scores = -f_sample / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
