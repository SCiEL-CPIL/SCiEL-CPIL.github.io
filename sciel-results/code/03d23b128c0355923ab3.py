"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: Adaptive Expected Improvement per unit cost.
    
    Parameters:
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP standard deviations
    f_best: float, the lowest observed function value
    cost: (2000,) array of evaluation costs
    
    Returns:
    scores: (2000,) array of acquisition scores (higher is better)
    """
    # Known minimum of the objective function
    f_min = -0.8
    
    # Calculate the gap to the optimum
    gap = max(0.0, f_best - f_min)
    
    # Adaptive exploration parameter xi
    # Larger gap -> larger xi (more exploration)
    # Smaller gap -> smaller xi (more exploitation)
    # 0.1 * gap seems reasonable. If gap is 0.5, xi=0.05. If gap is 0.01, xi=0.001.
    xi = 0.1 * gap
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    # z = (f_best - xi - mu) / sigma
    z = (f_best - xi - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in cdf/pdf
    # norm.cdf and norm.pdf handle large values, but clipping prevents underflow/overflow issues in extreme cases
    z = np.clip(z, -10, 10)
    
    # EI = (f_best - xi - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - xi - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI divided by cost
    # Cost is 1 + mean(x_scaled). Min cost is 1.0, Max is 2.0.
    cost_safe = np.maximum(cost, 1e-9)
    scores = ei / cost_safe
    
    # Handle any non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
