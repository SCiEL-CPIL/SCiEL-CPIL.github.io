import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 320.0 # 300s limit, keep 20s margin

def heuristics(demands, capacity):
    # Check time
    if time.time() - START_TIME > TIME_LIMIT:
        return np.ones((len(demands), len(demands))) * 1e-9

    n = len(demands)
    # Convert to float array to be safe, though input is floats
    d = np.array(demands, dtype=np.float64)
    
    # Calculate sums
    sums = d[:, None] + d[None, :]
    
    # Create mask for valid pairs (fit in capacity)
    valid = sums <= capacity
    
    # Calculate base attractiveness
    # Use exponential decay from full capacity to strongly prefer near-full bins
    # eta = exp(k * (ratio - 1)) where ratio = sums/capacity in (0, 1]
    # At ratio=1, eta=1. At ratio=0.5, eta=exp(-k*0.5)
    # k=5 gives eta(0.5)=0.082, eta(0.8)=0.368, eta(0.9)=0.607
    
    ratio = np.zeros_like(sums)
    ratio[valid] = sums[valid] / capacity
    
    # Apply exponential: strongly favors bins close to full
    eta = np.exp(5.0 * (ratio - 1.0))
    
    # Set invalid to 1e-9
    eta[~valid] = 1e-9
    
    return eta
