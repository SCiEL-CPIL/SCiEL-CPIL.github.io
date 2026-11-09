"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function with penalized exploration.
    
    Modifies standard Expected Improvement (EI) by reducing the weight of the
    uncertainty term (sigma * pdf(z)) to prioritize high-confidence improvements
    (exploitation) over risky exploration in the early sparse data regime.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Small epsilon to avoid division by zero
    eps = 1e-9
    sigma_safe = sigma + eps
    
    # Calculate Z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Exploitation component: Expected gain if mu < f_best
    exploit_term = (f_best - mu) * cdf
    
    # Exploration component: Value of information
    explore_term = sigma * pdf
    
    # Penalize exploration to force high-confidence improvements
    # Lambda < 1 reduces the influence of high-noise regions
    lambda_explore = 0.2
    
    # Combined Penalized EI
    ei_penalized = exploit_term + lambda_explore * explore_term
    
    # Cost-aware score: Higher is better
    # Divide by cost to account for evaluation expense
    # Cost is >= 1.0, so this is a mild normalization
    scores = ei_penalized / cost
    
    # Ensure all scores are finite
    # Replace NaN, Inf, -Inf with safe values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
