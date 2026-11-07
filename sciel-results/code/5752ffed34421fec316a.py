# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Diversity-aware acquisition function for cost-aware Bayesian optimization.
    
    Uses cost-normalized Expected Improvement (EI) and subtracts a logarithmic
    crowding penalty derived from GP uncertainty (sigma). Low sigma indicates
    well-explored (crowded) regions, which are penalized to encourage exploration
    of less certain areas, balancing exploitation and diversity.
    
    Args:
        mu: array of shape (2000,), GP mean predictions
        sigma: array of shape (2000,), GP standard deviation predictions
        f_best: scalar, best observed value so far
        cost: array of shape (2000,), evaluation cost for each candidate
    
    Returns:
        array of shape (2000,), acquisition scores (higher = evaluate next)
    """
    # 1. Compute Expected Improvement (EI)
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Cost-normalized EI
    cost_safe = np.maximum(cost, 1e-9)
    ei_cost = ei / cost_safe
    
    # 3. Calculate Crowding Penalty
    # Normalize sigma to [0, 1] using robust percentiles to handle outliers
    sigma_min = np.percentile(sigma, 1)
    sigma_max = np.percentile(sigma, 99)
    sigma_range = max(sigma_max - sigma_min, 1e-9)
    
    sigma_norm = (sigma - sigma_min) / sigma_range
    sigma_norm = np.clip(sigma_norm, 0.0, 1.0)
    
    # Crowding is high where sigma is low (well-explored)
    # crowding_score is 1 when sigma is min, 0 when sigma is max
    crowding_score = 1.0 - sigma_norm
    
    # Logarithmic penalty: gentle for moderate crowding, stronger for extreme crowding
    # log(1 + x) grows slowly, preventing the penalty from dominating completely
    # but ensuring low-sigma regions are discouraged.
    penalty_raw = np.log1p(crowding_score) # log(1 + crowding_score)
    
    # 4. Scale the penalty
    # We want the penalty to be comparable to the typical EI gains.
    # Use a high percentile of ei_cost to set the scale, ensuring the penalty
    # is significant relative to the best potential improvements.
    ei_scale = np.percentile(ei_cost, 95)
    if ei_scale <= 0:
        # If EI is generally non-positive, use a small fixed scale to encourage exploration
        ei_scale = 1e-3
        
    # The penalty should be subtracted. 
    # We apply a coefficient to tune the strength. 
    # A coefficient of 1.0 means the penalty can offset up to the 95th percentile of EI.
    # This strongly encourages moving to less crowded areas unless EI is exceptionally high.
    penalty = ei_scale * penalty_raw
    
    # 5. Final Score
    score = ei_cost - penalty
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
