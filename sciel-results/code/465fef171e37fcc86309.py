"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses a combination of Expected Improvement (EI) and a UCB-like term
    to balance exploitation and exploration. The key insight is that
    for 2D problems with a well-tuned GP, we can use a slightly more
    aggressive exploitation strategy while still maintaining some
    exploration to avoid getting stuck in local minima.
    
    mu: (N,) posterior mean
    sigma: (N,) posterior std
    f_best: best observed value so far
    
    Returns: (N,) scores, higher is better.
    """
    # Ensure finite values
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB term: encourages exploration in high-uncertainty regions
    # Use a moderate beta to balance exploitation and exploration
    beta = 0.5
    ucb = -mu + beta * sigma_safe
    
    # Combine EI and UCB with weights
    # Weight EI more heavily since we want to improve on f_best
    # But include UCB to explore uncertain regions
    score = 0.7 * ei + 0.3 * ucb
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
