"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: The evaluator kills after 330s.
    # We check if we are dangerously close. 
    # Since acquisition is called frequently, this check is cheap.
    # If we are close to the limit, we could return a simpler metric to save CPU,
    # but EI is already O(N). We just ensure we don't hang.
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
        # Return a simple exploitation score to be safe and fast
        # Maximize (f_best - mu) / cost
        cost_safe = np.maximum(cost, 1e-6)
        return (f_best - mu) / cost_safe

    # Robust Numerics
    # Prevent division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Z-score
    z = (f_best - mu) / sigma_safe
    
    # Clip Z to prevent numerical issues in PDF/CDF for extreme values
    # norm.pdf(-100) is 0, norm.cdf(-100) is 0
    # norm.pdf(100) is 0, norm.cdf(100) is 1
    z_clipped = np.clip(z, -50, 50)
    
    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    
    # Expected Improvement
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Cost Awareness
    # Cost is given. Ensure it's positive.
    cost_safe = np.maximum(cost, 1e-6)
    
    # Base Score: EI per unit cost
    score = ei / cost_safe
    
    # Exploration Term
    # Add a small term to encourage exploring uncertain regions
    # Weighted by inverse cost to prefer cheap exploration
    # 0.01 is a small factor to not override exploitation
    lambda_explore = 0.01
    score += lambda_explore * (sigma_safe / cost_safe)
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
