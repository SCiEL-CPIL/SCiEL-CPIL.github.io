import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = None
_DEADLINE = 300.0  # 300s as per task constraint, with safety margin

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Check deadline
    if time.time() - _start_time > _DEADLINE - 5.0:
        # Return uniform random scores to finish quickly
        return np.ones_like(mu) * 0.0
    
    # Calculate Expected Improvement
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip small negative values due to numerical precision
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    # cost is 1 + mean(scaled_x), so it's always >= 1
    base_score = ei / cost
    
    # Prior bias: The minimizer is in the inner 80% of [-4, 5]^4.
    # Range length is 9. Inner 80% is length 7.2.
    # The center of the original box [-4, 5] is 0.5.
    # The inner 80% region is symmetric around the center 0.5?
    # Let's check:
    # Original: [-4, 5]. Center = 0.5.
    # Inner 80%: from -4 + 0.1*9 = -3.1 to 5 - 0.1*9 = 4.1.
    # Center of inner region = (-3.1 + 4.1) / 2 = 0.5.
    # So the minimizer is uniformly distributed in a box centered at 0.5 in the original coordinates.
    
    # Coordinates are scaled to [0, 1] for the GP.
    # Original x in [-4, 5] -> scaled x in [0, 1].
    # x = -4 + 9 * x_scaled.
    # x = 0.5 -> 0.5 = -4 + 9 * x_scaled -> 4.5 = 9 * x_scaled -> x_scaled = 0.5.
    
    # So the expected location of the minimizer in scaled coordinates is 0.5.
    # Cost = 1 + mean(x_scaled).
    # Expected cost at the minimizer = 1 + 0.5 = 1.5.
    
    # We bias towards this expected cost.
    target_cost = 1.5
    
    # Gaussian weight on cost deviation
    # Lambda controls the strength of the bias.
    # A moderate lambda helps focus search without over-constraining.
    lambda_bias = 1.0
    weight = np.exp(-lambda_bias * (cost - target_cost) ** 2)
    
    # Add a small exploration bonus for high uncertainty to avoid premature convergence
    # This helps when EI is low but sigma is high, encouraging exploration of unexplored areas
    # The bonus is scaled by the inverse of cost to maintain cost-awareness
    exploration_bonus = sigma_safe / cost
    
    # Combine base score and exploration bonus
    # The exploration bonus is weighted to be smaller than the base score typically
    score = (base_score + 0.1 * exploration_bonus) * weight
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
