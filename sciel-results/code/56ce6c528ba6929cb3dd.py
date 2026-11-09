"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function with a global uncertainty penalty.
    
    Applies a strict exponential penalty to the entire Expected Improvement
    based on the standard deviation. This suppresses high-uncertainty regions
    more aggressively than penalizing only the exploration term, ensuring
    the search prioritizes high-confidence improvements.
    """
    # Small epsilon to avoid division by zero
    eps = 1e-9
    
    # Ensure sigma is positive for stability
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # z-score for standard normal CDF/PDF
    z = (f_best - mu) / sigma_safe
    
    # Standard EI components
    # Note: norm.cdf and norm.pdf handle arrays well
    ei_mean = (f_best - mu) * norm.cdf(z)
    ei_uncertainty = sigma_safe * norm.pdf(z)
    
    # Total Expected Improvement
    ei_total = ei_mean + ei_uncertainty
    
    # Strict Global Uncertainty Penalty:
    # Multiply the entire EI by exp(-alpha * sigma).
    # This ensures that if the model is uncertain (high sigma), the score
    # is heavily damped, regardless of the mean improvement estimate.
    # Alpha=2.5 provides a strong preference for low-noise regions.
    alpha = 2.5
    uncertainty_penalty = np.exp(-alpha * sigma_safe)
    
    # Penalized EI
    penalized_ei = ei_total * uncertainty_penalty
    
    # Ensure non-negative scores (numerical safety)
    penalized_ei = np.maximum(penalized_ei, 0.0)
    
    # Cost-aware: divide by cost to get improvement per unit cost
    scores = penalized_ei / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
