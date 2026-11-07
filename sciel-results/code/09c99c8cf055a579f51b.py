# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erf


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using probability-weighted improvement per unit cost.
    
    Implements: P(mu < f_best) * (f_best - mu) / cost
    Uses erf for faster and more stable probability calculation.
    """
    # Convert to numpy arrays for consistency
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate Phi(z) using erf: Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # sqrt(2) approx 1.41421356237
    prob_improve = 0.5 * (1.0 + erf(z / 1.41421356237))
    
    # Improvement magnitude: f_best - mu
    improvement_magnitude = f_best - mu
    
    # Probability-weighted improvement
    prob_weighted_improvement = prob_improve * improvement_magnitude
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate final score
    scores = prob_weighted_improvement / cost_safe
    
    # Sanitize: replace NaNs and Infs with 0.0
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
