import numpy as np
import time

# Global deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 320  # 320s out of 330s budget

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() > _DEADLINE:
        return -mu / (cost + 1e-9)
    
    # Ensure inputs are clean
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Structural Change: Simple UCB with Fixed Beta
    # Beta = 2.0 provides strong exploration to escape local minima
    # Cost penalty is linear to balance exploration with budget
    beta = 2.0
    score = (-mu + beta * sigma) / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
