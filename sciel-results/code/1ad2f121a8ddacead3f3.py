"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-adjusted Expected Improvement (EI) with exponential cost penalty.
    
    The acquisition score is EI(x) * exp(-lambda * cost(x)) where lambda
    is dynamically scaled by optimization progress. This creates a smooth
    tradeoff: early on, the penalty is mild allowing exploration; later,
    high-cost candidates are strongly penalized.
    
    mu, sigma: shape (2000,) GP mean and std in objective units
    f_best: lowest observed function value (scalar)
    cost: shape (2000,) evaluation cost for each candidate
    
    Returns: shape (2000,) acquisition scores (higher = evaluate next)
    """
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Dynamic lambda: scales with optimization progress
    # As f_best approaches 0 (the minimum), lambda increases,
    # making the cost penalty stronger
    # Use 1 / (1 + f_best) which ranges from ~1 (early) to large (late)
    # Clamp to avoid extreme values
    lambda_val = 1.0 / (1.0 + max(f_best, 0.0))
    # Scale lambda to be in a reasonable range
    lambda_val = np.clip(lambda_val, 0.1, 5.0)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Exponential cost penalty: exp(-lambda * cost)
    # Cost is in [1, 2], so lambda * cost is in [lambda, 2*lambda]
    # This creates a smooth, strictly decreasing penalty with cost
    cost_penalty = np.exp(-lambda_val * cost_safe)
    
    # Final score: EI * cost_penalty
    scores = ei * cost_penalty
    
    # Add a small exploration bonus for high-uncertainty regions
    # This prevents premature convergence by rewarding exploration
    # Scale the exploration bonus by the same cost penalty
    exploration_bonus = 0.01 * sigma_safe * cost_penalty
    scores = scores + exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check (computation is O(N), should be very fast)
    if time.time() - start_time > 280.0:
        return np.zeros_like(mu)
        
    return scores
# EVOLVE-BLOCK-END
