import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) divided by cost.
    
    Args:
        mu: array of shape (2000,), mean predictions.
        sigma: array of shape (2000,), standard deviation predictions.
        f_best: float, best observed value.
        cost: array of shape (2000,), cost of evaluating each point.
        
    Returns:
        array of shape (2000,), scores (higher is better).
    """
    start_time = time.time()
    # Safety margin: finish within 290s of start
    deadline = start_time + 290.0
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle potential numerical issues
    # Clip sigma to a small positive value to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Clip cost to a small positive value
    cost_safe = np.maximum(cost, 1e-6)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # Phi is the standard normal CDF, phi is the standard normal PDF
    
    # Calculate z-scores
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF
    # norm.cdf and norm.pdf are vectorized
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Calculate EI
    # Note: (f_best - mu) can be negative if mu > f_best, in which case EI should be 0 or negative?
    # Standard EI formula handles this: if mu > f_best, z is negative.
    # (f_best - mu) is negative. Phi(z) is small. sigma * pdf(z) is positive.
    # Usually EI is clipped to be non-negative for acquisition, but for ranking, 
    # the raw value works as long as it's monotonic.
    # However, if mu > f_best, the expected improvement is technically 0 (you can't improve on the best).
    # The formula (f_best - mu) * Phi(z) + sigma * phi(z) yields a small positive number even when mu > f_best 
    # due to the sigma * phi(z) term? No, let's check.
    # If mu = f_best + delta (delta > 0), z = -delta/sigma.
    # Term 1: -delta * Phi(-delta/sigma). Small negative.
    # Term 2: sigma * phi(-delta/sigma). Positive.
    # The sum is the expected value of max(f_best - Y, 0). It is always non-negative.
    
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    score = ei / cost_safe
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Check deadline
    if time.time() > deadline:
        pass
        
    return score
