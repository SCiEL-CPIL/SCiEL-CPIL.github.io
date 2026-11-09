import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a hybrid approach combining Expected Improvement (EI) with cost normalization
    and a small exploration bonus to balance exploitation and exploration.
    
    Parameters:
    - mu: array of shape (2000,) with GP mean predictions
    - sigma: array of shape (2000,) with GP standard deviation predictions
    - f_best: float, the best (lowest) observed value so far
    - cost: array of shape (2000,) with evaluation costs
    
    Returns:
    - array of shape (2000,) with acquisition scores (higher = better)
    """
    # Ensure sigma is non-negative and add a small epsilon to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-scores: how many standard deviations below f_best the mean is
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (it should be, but clip for safety)
    ei = np.maximum(ei, 0)
    
    # Cost normalization: divide by cost to get "improvement per unit cost"
    # Add a small epsilon to cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-8)
    cost_aware_ei = ei / cost_safe
    
    # Add a small exploration bonus based on uncertainty
    # This helps escape local minima in the multimodal Levy landscape
    # The bonus is scaled by the relative uncertainty compared to the best improvement seen
    # Use a fixed exploration weight that balances exploitation and exploration
    exploration_bonus = 0.1 * sigma_safe / cost_safe
    
    # Final score: combine cost-aware EI with exploration bonus
    score = cost_aware_ei + exploration_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
