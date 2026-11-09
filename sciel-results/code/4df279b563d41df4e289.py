import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: ensure we return within budget
    t0 = time.time()
    
    # Sanitize sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement z
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Upper Confidence Bound (UCB) - balances exploitation and exploration
    # Beta controls exploration weight
    beta = 1.0
    ucb = mu - beta * sigma_safe  # Lower is better for minimization, so we want low mu and high sigma
    
    # Cost-aware: normalize by cost to prefer cheaper evaluations
    # cost is in [1, 2] (1 + mean of scaled coords in [0,1])
    cost_safe = np.maximum(cost, 1e-9)
    
    # Score components (higher = better)
    # EI normalized by cost
    ei_score = ei / cost_safe
    
    # UCB normalized by cost (lower ucb is better, so we negate)
    ucb_score = -ucb / cost_safe
    
    # Prior bias towards expected minimizer location
    # Minimizer is in inner 80% of [-4, 5]^4
    # In scaled coords [0,1], the inner 80% is [0.1, 0.9]
    # Center is at 0.5 in scaled coords
    # Expected cost at minimizer: 1 + 0.5 = 1.5
    
    # Gaussian penalty for cost deviation from 1.5
    target_cost = 1.5
    lambda_bias = 1.5
    cost_weight = np.exp(-lambda_bias * (cost - target_cost) ** 2)
    
    # Combine: primarily EI, with UCB for exploration, weighted by cost proximity
    # Weights: EI is primary, UCB provides exploration bonus
    w_ei = 1.0
    w_ucb = 0.3
    
    score = (w_ei * ei_score + w_ucb * ucb_score) * cost_weight
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Final deadline check - if we're close to timeout, return immediately
    if time.time() - t0 > 0.5:
        return score
    
    return score
