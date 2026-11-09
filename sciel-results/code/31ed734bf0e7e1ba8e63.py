"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses a robust combination of Expected Improvement and UCB with
    rank-based normalization for outlier resistance.
    
    mu, sigma: arrays of shape (2000,) - posterior mean and std in objective units
    f_best: float - lowest observed value so far
    
    Returns: array of shape (2000,) - scores, higher means evaluate next
    """
    start_time = time.time()
    DEADLINE = start_time + 295.0  # Safety margin before 330s limit
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = np.maximum(sigma, 1e-12)  # Avoid division by zero
    
    # Clip extreme values for numerical stability
    mu = np.clip(mu, f_best - 50, f_best + 50)
    sigma = np.clip(sigma, 1e-10, 100)
    
    # Compute z-scores
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -10, 10)
    
    # Base Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB component with beta tuned for multi-modal landscape
    beta = 2.0
    
    ucb = mu + beta * sigma
    # Negate because we want to minimize (lower mu is better)
    ucb_score = -ucb
    
    # Weights: slightly UCB-heavy to encourage basin exploration
    w_ei = 0.55
    w_ucb = 0.45
    
    # Rank-based normalization for robustness to outliers
    n = len(ei)
    ei_ranks = np.argsort(np.argsort(ei)) / max(n - 1, 1)
    ucb_ranks = np.argsort(np.argsort(ucb_score)) / max(n - 1, 1)
    
    # Combined score
    score = w_ei * ei_ranks + w_ucb * ucb_ranks
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() > DEADLINE:
        # Fallback: return pure EI if we're running out of time
        return np.where(np.isfinite(ei), ei, 0.0)
    
    return score
# EVOLVE-BLOCK-END
