"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining PI (Probability of Improvement) 
    and EI (Expected Improvement) with variance penalty and cost normalization.
    
    Parameters:
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP standard deviations
    f_best: float, the lowest observed function value
    cost: (2000,) array of evaluation costs
    
    Returns:
    scores: (2000,) array of acquisition scores (higher is better)
    """
    start_time = time.time()
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # PI = Phi(z)
    pi = norm.cdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * pi + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Variance penalty to reduce sensitivity to noise
    variance_penalty = 0.1 * sigma_safe
    
    # Combine PI and EI with variance penalty
    combined = 0.5 * pi + 0.5 * ei - variance_penalty
    combined = np.maximum(combined, 0.0)
    
    # Cost-aware score
    cost_safe = np.maximum(cost, 1e-9)
    scores = combined / cost_safe
    
    # Handle any non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard: if we're taking too long, return early
    elapsed = time.time() - start_time
    if elapsed > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
