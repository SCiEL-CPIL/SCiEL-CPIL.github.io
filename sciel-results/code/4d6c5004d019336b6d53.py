"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a multiplicative PI variant.
    
    The score combines exploitation (expected improvement) with exploration
    (uncertainty) in a way that is cost-aware and prevents wasted exploration
    in hopeless regions.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma = np.clip(sigma, 1e-9, None)
    
    # Clip cost to avoid division by zero
    cost = np.clip(cost, 1e-9, None)
    
    # Standard normal CDF and PDF
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI is non-negative
    
    # Multiplicative PI: PI * (1 + beta * sigma)
    # This couples exploitation and exploration more tightly
    # beta controls the exploration strength
    beta = 1.0
    
    # The multiplicative form prevents wasted exploration in hopeless regions
    # while boosting uncertain regions with any improvement chance
    score = ei * (1.0 + beta * sigma)
    
    # Divide by cost for cost-awareness
    score = score / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
