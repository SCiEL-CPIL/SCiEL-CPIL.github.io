"""Cost-aware acquisition for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

# Pre-compute constants for numerical stability
_EPS = 1e-9


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Expected Improvement
    with an exploration term (UCB-like).
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, _EPS)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, _EPS)
    
    sigma_safe = np.maximum(sigma, _EPS)
    
    # 1. Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Exploration term (UCB style)
    # We add a term proportional to uncertainty to encourage exploration.
    # A fixed beta is often robust. We use beta=1.0 as a starting point.
    # The cost adjustment handles the trade-off: high cost reduces the score.
    beta = 1.0
    exploration = beta * sigma_safe
    
    # 3. Combine EI and Exploration
    # Score = (EI + Exploration) / Cost
    # This directly optimizes the expected gain per unit cost.
    score = (ei + exploration) / cost
    
    # Ensure all scores are finite and non-negative (though EI is >=0, exploration is >=0, cost >0)
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
