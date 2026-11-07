"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a hybrid approach: Probability of Improvement (POI) combined with
    Expected Improvement (EI), both normalized by cost. This balances exploration
    and exploitation while being aware of evaluation costs.
    """
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute z-score for improvement
    z = (f_best - mu) / sigma
    
    # Probability of Improvement: P(f(x) < f_best)
    poi = norm.cdf(z)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize EI by the scale of f_best to make it dimensionless
    # This prevents EI from dominating when f_best is large
    scale = max(abs(f_best), 1e-6)
    ei_normalized = ei / scale
    
    # Combine POI and normalized EI with a weight
    # POI provides exploration, EI provides exploitation
    # Weight POI more heavily to encourage exploration in the early stages
    combined = 0.7 * poi + 0.3 * ei_normalized
    
    # Normalize by cost to get "improvement per unit cost"
    scores = combined / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
