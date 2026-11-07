"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition score for 2000 candidates. Higher score = evaluate next.
    
    Uses a hybrid of Probability of Improvement and a budget-aware
    exploration bonus with soft cost weighting.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    # Probability of Improvement: P(f(x) < f_best)
    z = (f_best - mu) / sigma
    pi = norm.cdf(z)
    
    # Budget-aware exploration bonus
    # cost is the total cost consumed so far; budget is 30
    # As cost approaches 30, remaining budget shrinks, so reduce exploration
    budget = 30.0
    remaining_ratio = np.clip(1.0 - cost / budget, 0.0, 1.0)
    
    # Beta: high exploration early, low exploration late
    # Use a smooth schedule: beta goes from ~1.0 to ~0.1
    beta = 0.1 + 0.9 * remaining_ratio
    
    # Exploration bonus: beta * sigma (uncertainty-driven)
    exploration_bonus = beta * sigma
    
    # Combine: PI + exploration bonus
    score = pi + exploration_bonus
    
    # Soft cost weighting: prefer cheaper points gently
    # Use (1 + 1/(1+cost)) which ranges from ~1.5 (cost~0) to ~1.0 (cost>>1)
    # This is much more stable than dividing by cost
    cost_weight = 1.0 + 1.0 / (1.0 + cost)
    score = score * cost_weight
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
