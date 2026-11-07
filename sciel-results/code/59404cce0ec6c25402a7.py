import numpy as np

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing algorithm.
    
    Strategy:
    We favor pairs (i, j) that:
    1. Fit within the capacity.
    2. Have a large combined demand (d_i + d_j), as larger items are harder to place.
    3. Have low waste (capacity - d_i - d_j), as tight fits reduce remaining capacity.
    
    Formula:
    eta[i, j] = (d_i + d_j) * exp(-waste / sigma)
    where waste = capacity - (d_i + d_j)
    and sigma is a scale parameter for the exponential decay (set to capacity/10).
    
    This strongly rewards tight fits of large items, guiding the ants to pack bins
    efficiently and close them quickly.
    """
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Compute pairwise sums
    pair_sums = demands[:, None] + demands[None, :]
    
    # Determine which pairs fit in the bin
    fits = pair_sums <= capacity
    
    # Compute waste for fitting pairs
    waste = capacity - pair_sums
    
    # Initialize heuristic matrix with a small positive value for non-fitting pairs
    eta = np.full((n, n), 1e-9, dtype=float)
    
    if np.any(fits):
        # Scale parameter for exponential decay
        # Smaller sigma means sharper preference for tight fits
        sigma = capacity / 10.0
        
        # Compute exponential decay based on waste
        # waste is 0 for perfect fit, so exp(0) = 1
        # waste is large for loose fits, so exp(-large) ~ 0
        decay = np.exp(-waste / sigma)
        
        # Heuristic: Combined demand * Tightness
        # We want to broadcast pair_sums (n, n) and decay (n, n)
        # Only apply to fitting pairs
        
        # Calculate the raw heuristic for all pairs, then mask
        # Note: pair_sums and decay are (n, n)
        raw_eta = pair_sums * decay
        
        # Apply mask: only fitting pairs get the calculated value
        # Non-fitting pairs remain 1e-9
        eta[fits] = raw_eta[fits]
        
        # Safety: ensure no values are 0 or negative (exp is > 0, sums > 0)
        # But if any are 0 due to underflow, replace with 1e-9
        eta[eta == 0] = 1e-9
        
    return eta
