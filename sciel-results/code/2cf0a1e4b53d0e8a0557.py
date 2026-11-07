import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI).
    
    Args:
        mu: Array of mean predictions (shape 2000).
        sigma: Array of standard deviations (shape 2000).
        f_best: Best observed value so far.
        cost: Array of costs for each candidate (shape 2000).
        
    Returns:
        Array of scores (shape 2000). Higher scores indicate better candidates.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate the z-score
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    phi_z_cdf = norm.cdf(z)
    
    ei = (f_best - mu) * phi_z_cdf + sigma * phi_z
    
    # EI can be negative if mu > f_best (which is expected for most points), 
    # but mathematically EI is defined as E[max(f_best - f(x), 0)], so it should be >= 0.
    # The formula above yields negative values when the probability of improvement is very low 
    # and the term (f_best - mu) is large negative. 
    # Let's clamp EI to be non-negative to strictly follow the definition of "Improvement".
    ei = np.maximum(ei, 0.0)
    
    # Weight by inverse cost to favor cheaper evaluations
    # Cost is in range [1, 2], so 1/cost is in [0.5, 1]
    # We want higher scores for better candidates.
    scores = ei / cost
    
    # Handle any potential numerical issues (NaN, Inf)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
