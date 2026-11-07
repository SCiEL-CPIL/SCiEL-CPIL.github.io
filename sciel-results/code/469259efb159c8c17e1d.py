"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for Expected Improvement
    # Use a small xi (improvement threshold) to encourage more significant jumps
    xi = 0.01
    z = (f_best - xi - mu) / sigma_safe
    
    # Expected Improvement with xi
    ei = (f_best - xi - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring
    # Using EI / cost is standard. 
    # To slightly boost exploration in early stages or if EI is flat, 
    # we could blend with UCB, but let's stick to EI/cost with xi.
    scores = ei / cost
    
    # Return finite scores
    return scores
# EVOLVE-BLOCK-END
