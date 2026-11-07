"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement per unit Cost (EI/Cost).
    
    This structural change moves from an additive cost penalty (EI - lambda*cost) 
    to a ratio-based metric (EI / cost). This explicitly optimizes for the 
    "efficiency" of the evaluation, seeking the highest expected improvement 
    per unit of cost spent. This tends to favor cheaper exploratory moves 
    early on and only selects expensive moves when the expected improvement 
    is substantial, leading to better budget utilization.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    # Since we are minimizing, improvement is f_best - mu (positive if mu < f_best)
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = diff * cdf_z + sigma * pdf_z
    
    # Ensure finiteness and non-negativity
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware weighting: EI divided by Cost
    # We add a small epsilon to cost to avoid division by zero, though cost >= 1.
    cost_safe = np.maximum(cost, 1e-9)
    
    # Score = EI / Cost
    # This maximizes improvement efficiency.
    score = ei / cost_safe
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
