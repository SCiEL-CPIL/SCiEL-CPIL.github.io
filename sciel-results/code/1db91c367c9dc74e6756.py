# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    and Lower Confidence Bound (LCB) to balance exploitation and exploration.
    
    The score is a weighted combination of:
    1. EI / cost: Exploitation - expected improvement per unit cost
    2. LCB / cost: Exploration - uncertainty per unit cost
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Best observed value so far
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Safety: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost-aware Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For numerical stability, use scipy's norm.cdf and norm.pdf
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Lower Confidence Bound for exploration
    # LCB = mu - beta * sigma
    # We use a moderate beta to encourage exploration
    beta = 2.0
    lcb = mu - beta * sigma_safe
    
    # Cost-aware scores
    cost_safe = np.maximum(cost, 1e-6)
    
    # Weighted combination: 0.7 * EI/cost + 0.3 * (-LCB)/cost
    # Note: LCB is lower for more promising points, so we negate it
    # Actually, for minimization, LCB = mu - beta*sigma is the lower bound
    # We want to maximize the score, so we use -LCB or just LCB with appropriate sign
    # Since we're minimizing, lower LCB is better, so we use -LCB as the exploration term
    exploration_term = -lcb  # Higher when LCB is lower (more promising)
    
    # Combine exploitation and exploration
    scores = (0.7 * ei + 0.3 * exploration_term) / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
