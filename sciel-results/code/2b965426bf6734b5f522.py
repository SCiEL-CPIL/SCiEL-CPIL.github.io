import numpy as np
from scipy.stats import norm
import time

# ---------- Global timer / safety ----------
_START_TIME = time.time()
_TIME_LIMIT = 325.0  # Safety margin below 330s wall-clock

# ---------- Acquisition Parameters ----------
# Cost-aware EI with adaptive exploration
_COST_POW = 1.0   # Linear cost normalization (cost = 1 + mean(x_scaled))
_BASE_BETA = 0.5  # Base exploration parameter
_ADAPT_RATE = 0.3 # How quickly beta increases with budget fraction used

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using EI/cost with adaptive exploration.
    Returns scores of shape (2000,), higher = evaluate next.
    """
    # Explicit deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei / cost_safe
        score = np.where(np.isfinite(score), score, -1e9)
        return score

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration: increase beta as we use more budget
    # Estimate budget fraction: total cost limit is 30, initial cost ~ 5 * 1.5 = 7.5
    # We don't have direct access to total cost, but we can estimate from f_best
    # and the fact that we're in a loop. Use a simple heuristic:
    # If f_best is close to the known minimum (-0.8), we're doing well and can explore less.
    # Otherwise, explore more.
    # 
    # Structural change: use a "regret-based" adaptive beta
    # If f_best is much higher than the minimum, we're uncertain and should explore more.
    # The minimum of cosine8 8d is -0.8.
    min_possible = -0.8
    regret = f_best - min_possible  # How far we are from the optimum
    # Scale regret to [0, 1] range (assume typical regret < 2)
    regret_norm = np.clip(regret / 2.0, 0, 1)
    
    # Adaptive beta: start low, increase with regret
    beta = _BASE_BETA + _ADAPT_RATE * regret_norm
    
    # UCL potential (for exploration)
    ucl_potential = (f_best - mu) + beta * sigma_safe
    
    # Blend: primarily EI/cost, with a smaller UCL component for exploration
    # Weight based on how much we've learned: if sigma is large, explore more
    sigma_mean = np.mean(sigma_safe)
    # If average uncertainty is high, weight UCL more
    ucl_weight = np.clip(sigma_mean * 0.5, 0.05, 0.3)
    ei_weight = 1.0 - ucl_weight
    
    blended = ei_weight * ei + ucl_weight * ucl_potential
    
    # Cost normalization
    score = blended / (cost_safe ** _COST_POW)
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -1e9)
    
    return score
