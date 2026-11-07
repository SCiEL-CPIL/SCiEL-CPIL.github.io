"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_deadline = _start_time + 290.0

def acquisition(mu, sigma, f_best):
    # Deadline guard: if we are running out of time, fall back to greedy
    if time.time() > _deadline:
        return -mu

    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # We use a hybrid acquisition: UCB with a beta that balances exploration and exploitation.
    # For Ackley 2D with 30 evaluations, we need to explore enough to find the global min at (0,0).
    # UCB score = -mu + beta * sigma
    # But we also want to ensure we don't ignore very promising points (low mu).
    # 
    # Alternative: Use a weighted combination of EI and UCB, or use qEI.
    # Given the limited budget (30 evals), pure UCB with beta=1 was shown to improve score.
    # Let's try a slightly more sophisticated approach:
    # 
    # Use the "Expected Improvement" but with a gamma parameter to encourage exploration.
    # Or use the "Probability of Improvement" with a threshold.
    # 
    # Actually, let's stick with UCB but tune beta. The inspiration showed beta=1 works well.
    # Let's try beta = 0.5 + some function of sigma to encourage more exploration in high-uncertainty regions.
    # Or simply use a fixed beta that's slightly less than 1 to balance exploration and exploitation.
    
    # Let's use a beta that starts high and decreases, but we don't have access to step count.
    # So we'll use a fixed beta. Let's try beta = 0.7.
    
    # Score = -mu + beta * sigma
    beta = 0.7
    score = -mu + beta * sigma
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
