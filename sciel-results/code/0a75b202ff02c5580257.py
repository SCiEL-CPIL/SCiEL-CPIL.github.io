"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global variable to track time for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: If we are getting close to the limit, return a simple fallback
    # The limit is 300s. We assume the loop is short, but if this function is called
    # repeatedly, we check elapsed time.
    # However, 2000 points is fast. We'll just ensure we don't hang.
    
    # Use UCB (Upper Confidence Bound) for minimization.
    # We want to minimize f(x).
    # LCB = mu - beta * sigma. We want to minimize LCB.
    # So we maximize -LCB = beta * sigma - mu.
    # To incorporate f_best, we look at the potential improvement:
    # Score = (f_best - mu + beta * sigma) / cost_term
    
    # Hypothesis: 
    # 1. UCB is better for multimodal functions (Griewank) than EI.
    # 2. A softer cost penalty (1/sqrt(cost)) prevents over-penalizing the 
    #    center of the domain (cost ~1.5) vs corners (cost ~1.0).
    
    beta = 2.5
    # Ensure cost is at least 1 to avoid division by zero or extreme values
    cost_safe = np.maximum(cost, 1.0)
    
    # Soft cost penalty
    cost_term = np.sqrt(cost_safe)
    
    # UCB Score for minimization
    # We want high scores to be selected.
    # If mu < f_best, (f_best - mu) is positive.
    # If sigma is high, we add a bonus.
    score = (f_best - mu + beta * sigma) / cost_term
    
    # Handle numerical issues
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
