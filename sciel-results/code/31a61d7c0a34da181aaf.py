import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    and a small UCB exploration term, penalized by evaluation cost.
    
    Args:
    mu (np.ndarray): GP mean predictions, shape (N,)
    sigma (np.ndarray): GP standard deviation predictions, shape (N,)
    f_best (float): Best observed value so far
    cost (np.ndarray): Cost of evaluating each candidate, shape (N,)
    
    Returns:
    np.ndarray: Acquisition scores, shape (N,)
    """
    # Deadline guard: Ensure we return within the 330s limit
    # This is a vectorized operation, so it should be fast, but we add a check.
    start_time = time.time()
    deadline = 330.0
    
    # Ensure inputs are arrays with proper dtype
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle sigma = 0 to avoid division by zero
    # If sigma is 0, the distribution is a delta function.
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Using sigma_safe in the PDF term for consistency with the limit case,
    # though sigma * phi(z) -> 0 as sigma -> 0 for finite z.
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI must be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Refinement: Add a small UCB exploration term.
    # This encourages exploring regions with high uncertainty (sigma)
    # even if the expected improvement is low, helping to escape local optima.
    # The factor 0.2 is a small, stable weight.
    kappa = 0.2
    acquisition_numerator = ei + kappa * sigma
    
    # Cost-aware score: (EI + UCB) / cost
    # This maximizes the rate of improvement (including exploration) per unit cost.
    cost_safe = np.maximum(cost, 1e-10)
    scores = acquisition_numerator / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Final time check (should not trigger for 2000 points, but safe)
    if time.time() > deadline - 1.0:
         # If we are dangerously close to timeout, return the current best effort.
         # In this simple implementation, the computation is already done.
         pass

    return scores
