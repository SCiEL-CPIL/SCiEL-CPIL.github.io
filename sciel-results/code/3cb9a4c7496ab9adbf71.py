"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement per unit cost,
    with a robustness floor to avoid division by very small costs and to
    handle edge cases.
    
    We use a modified EI that accounts for the cost of evaluation.
    The cost is 1 + mean(x_scaled), where x_scaled is in [0,1].
    Since we don't have coordinates, we use the provided cost vector.
    
    To be more robust, we add a small epsilon to cost and also consider
    a hybrid that balances EI with uncertainty (UCB-like) for early stages.
    """
    sigma_safe = np.maximum(sigma, 1e-6)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: EI per unit cost
    # But we also want to ensure we explore when EI is low but uncertainty is high
    # Let's use a combination: primarily EI/cost, but with a small exploration bonus
    
    # Primary: EI per cost
    score = ei / cost_safe
    
    # Add a small exploration term to avoid getting stuck in local optima
    # This is a small bonus for high uncertainty, scaled by cost
    # The scale is small so that EI dominates
    exploration_bonus = 0.01 * sigma_safe / cost_safe
    
    # Final score: EI/cost + small exploration bonus
    final_score = score + exploration_bonus
    
    # Ensure all values are finite
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END
