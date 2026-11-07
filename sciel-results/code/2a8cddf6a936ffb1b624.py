"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a combination of Expected Improvement and UCB with cost-awareness.
    The cost structure favors cheaper regions (lower cost = better value for money).
    """
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB term - exploration
    # Use a moderate beta for exploration
    beta = 0.5
    ucb = mu + beta * sigma_safe
    
    # Normalize cost to [0, 1] range for scaling
    # cost is typically in [1, 2]
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min + 1e-9
    cost_norm = (cost - cost_min) / cost_range  # 0 = cheapest, 1 = most expensive
    
    # Cost-aware weighting: penalize expensive points more
    # Use exponential decay to strongly favor cheap regions
    cost_penalty = np.exp(-1.0 * cost_norm)
    
    # Combine EI and UCB with cost-awareness
    # EI is in objective units (positive = improvement)
    # UCB is in objective units (lower = better for minimization)
    
    # For minimization, we want low mu and high uncertainty
    # EI already captures this; UCB gives exploration bonus
    
    # Scale EI to be comparable with cost penalty
    # Use EI per unit cost as primary signal
    ei_per_cost = ei / cost
    
    # Add a small UCB component for exploration, also cost-aware
    # For minimization, negative UCB is good (low predicted value)
    # ucb_explore = -ucb / cost  (negative of UCB per cost, higher is better)
    
    # Actually, let's think differently:
    # Primary: EI per cost (exploitation)
    # Secondary: exploration bonus for high uncertainty in cheap regions
    
    # Exploration bonus: high sigma in cheap regions
    explore_bonus = sigma_safe * cost_penalty * 0.1
    
    # Final score: combine EI/cost with exploration bonus
    # Make sure all values are finite
    score = ei_per_cost + explore_bonus
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
