import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using expected improvement per unit cost.
    
    We compute the expected improvement for each candidate and divide by the
    evaluation cost to get a cost-aware score. This encourages selecting
    points that offer high expected improvement relative to their cost.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    # sigma can be very small; avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf and norm.pdf
    z_clipped = np.clip(z, -50, 50)
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_val = norm.cdf(z_clipped)
    pdf_val = norm.pdf(z_clipped)
    
    ei = (f_best - mu) * cdf_val + sigma * pdf_val
    
    # Cost-aware score: EI per unit cost
    # This naturally balances exploration (high EI) with cost efficiency
    scores = ei / cost_safe
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
