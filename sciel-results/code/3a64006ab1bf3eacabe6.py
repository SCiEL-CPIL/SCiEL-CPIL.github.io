"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Powell 4D.
    
    Parameters:
        mu: (2000,) array of GP mean predictions (standardized units)
        sigma: (2000,) array of GP std predictions (standardized units)
        f_best: scalar, best observed value (standardized units)
        cost: (2000,) array of evaluation costs (in [1, 2])
    
    Returns:
        (2000,) array of acquisition scores (higher = more desirable)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Small UCB exploration bonus (for minimization: -mu + kappa*sigma)
    kappa = 0.15
    ucb = -mu + kappa * sigma_safe
    
    # Combine: primarily EI, with a small UCB component to encourage exploration
    # Scale UCB to be comparable to EI magnitude
    # Use a weighted combination
    score = (ei + 0.1 * np.maximum(ucb, 0.0)) / cost_safe
    
    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
