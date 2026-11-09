import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
        # Ensure cost is not zero
        cost = np.maximum(cost, 1e-6)
        
        # 1. Expected Improvement (EI)
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        
        # 2. Upper Confidence Bound (UCB) for minimization
        # We want to maximize the potential for a low value.
        # UCB_min = -mu + beta * sigma
        # Choose beta adaptively or fixed. Let's use 2.0.
        beta = 2.0
        ucb = -mu + beta * sigma
        
        # Normalize UCB to be on a similar scale to EI?
        # EI is in objective units. UCB is in objective units.
        # However, EI is always >= 0. UCB can be negative.
        # We want higher scores to be better.
        
        # Cost-aware versions
        ei_cost = ei / cost
        ucb_cost = ucb / cost
        
        # Combine
        # We need to balance them.
        # Let's weight them equally.
        score = 0.5 * ei_cost + 0.5 * ucb_cost
        
        # Ensure finite
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        
        return score
