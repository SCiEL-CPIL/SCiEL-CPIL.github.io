"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function with penalized Expected Improvement (EI).
    
    Args:
        mu: Array of shape (N,) with predicted means.
        sigma: Array of shape (N,) with predicted standard deviations.
        f_best: Scalar, the best observed value so far.
        cost: Array of shape (N,) with evaluation costs.
        
    Returns:
        Array of shape (N,) with acquisition scores (higher is better).
    """
    epsilon = 1e-8
    sigma_safe = np.maximum(sigma, epsilon)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a smooth penalty term that grows with uncertainty to suppress
    # high-noise regions in the early sparse-data regime
    # This forces the search to prioritize high-confidence improvements
    penalty_strength = 0.5
    penalty = penalty_strength * sigma_safe * np.exp(-sigma_safe)
    
    # Combine EI with penalty: reward improvement potential but penalize uncertainty
    penalized_ei = ei_scores - penalty
    
    # Make cost-aware by dividing by cost
    cost_safe = cost + epsilon
    
    scores = penalized_ei / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
