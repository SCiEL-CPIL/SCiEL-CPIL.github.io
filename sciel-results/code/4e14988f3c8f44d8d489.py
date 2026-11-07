"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using marginal Expected Improvement per unit cost.
    
    This treats the budget as a constrained resource and selects the candidate
    that maximizes the rate of expected improvement (dEI/dcost), which naturally
    balances exploration, exploitation, and evaluation cost.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    start_time = time.time()
    deadline = 295.0  # Safety margin before 300s per-eval limit and 330s total
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI is non-negative by construction for minimization
    ei = np.maximum(ei, 0.0)
    
    # Cost is in [1, 2] range (1 + mean(x_scaled_to_0_1))
    # The marginal improvement per unit cost is EI / cost
    # This naturally prioritizes:
    #   - High EI candidates (good improvement expected)
    #   - Low cost candidates (cheaper to evaluate)
    # This is structurally different from multiplicative penalties or fixed-weight blends.
    
    # Use a small epsilon to avoid division by zero
    cost_safe = np.maximum(cost, 1e-8)
    
    # Primary score: marginal EI per unit cost
    marginal_ei = ei / cost_safe
    
    # Add a small exploration bonus scaled by uncertainty, also cost-normalized
    # This helps escape local optima in the GP model
    # The bonus is small to not overwhelm the EI signal
    exploration_bonus = sigma_safe / cost_safe * 0.1
    
    # Combine: marginal EI dominates, exploration bonus is secondary
    scores = marginal_ei + exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() - start_time > deadline:
        # Return what we have (should be fast enough normally)
        return scores
    
    return scores
# EVOLVE-BLOCK-END
