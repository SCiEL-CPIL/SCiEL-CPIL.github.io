import numpy as np

def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Pair sums
    pair_sums = demands[:, None] + demands[None, :]
    
    # Validity mask
    valid = pair_sums <= capacity
    
    # Heuristic values
    # We want to reward tight fits.
    # Let's use a combination of the sum ratio and inverse waste.
    # Or just inverse waste.
    
    # Initialize with 0
    H = np.zeros((n, n), dtype=float)
    
    # Calculate for valid pairs
    # Waste = C - S
    waste = capacity - pair_sums
    
    # For valid pairs, waste >= 0.
    # To avoid div by 0, add epsilon.
    # We use 1.0 as epsilon to keep values <= 1.0.
    # If waste=0, val=1.
    # If waste=1, val=0.5.
    # If waste=10, val=0.09.
    
    # This creates a very steep preference for tight fits.
    # Let's try a softer version: 1 / (waste + 10)
    # Waste 0 -> 0.1
    # Waste 10 -> 0.05
    # Waste 100 -> 0.009
    # Ratio 100:1. Still strong.
    
    # Let's stick to 1 / (waste + 1) as it maximizes the distinction between "perfect" fit and "bad" fit.
    
    H[valid] = 1.0 / (waste[valid] + 1.0)
    
    # Zero out diagonal to prevent self-placement if not handled by solver
    np.fill_diagonal(H, 0.0)
    
    return H
