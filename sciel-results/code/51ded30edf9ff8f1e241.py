import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing algorithm.
    
    The heuristic eta[i, j] represents the attractiveness of placing item j
    immediately after item i in the same bin.
    
    Strategy:
    Composite heuristic that balances:
    1. Tight fit: minimize remaining gap after placing i+j
    2. Item size: prefer larger items to reduce stranded large items
    3. Bin utilization: reward pairs that fill a high fraction of capacity
    4. Smooth surrogate for best-fit using log-sum-exp style weighting
    
    Formula: eta[i, j] = item_size_j * exp(-alpha * gap) * (1 + beta * utilization)
    where gap = capacity - w_i - w_j
    """
    W = np.array(demands, dtype=float)
    n = len(W)
    
    # Calculate sums of all pairs
    pair_sums = W[:, None] + W[None, :]
    
    # Calculate gaps
    gaps = capacity - pair_sums
    
    # Identify valid pairs
    fits = gaps >= 0
    
    # Initialize heuristic matrix with a very small value for invalid pairs
    H = np.full((n, n), 1e-9)
    
    if np.any(fits):
        # Parameters tuned for balance between tight-fit and item-size preferences
        alpha = 0.05  # Controls how strongly we penalize large gaps
        beta = 2.0    # Controls how strongly we reward high utilization
        
        # Exponential decay of attractiveness with gap size
        # This creates a smooth preference for tight fits
        gap_penalty = np.exp(-alpha * gaps)
        
        # Utilization factor: how much of the bin is filled by i+j
        utilization = pair_sums / capacity
        
        # Item size factor: prefer placing larger items
        # W[None, :] broadcasts to (n, n) for column j
        item_size = W[None, :]
        
        # Composite: reward tight fits (exponential decay) and large items
        # The (1 + beta * utilization) term adds a smooth boost for well-filled bins
        composite = item_size * gap_penalty * (1.0 + beta * utilization)
        
        # Set invalid pairs to 1e-9
        H = np.where(fits, composite, 1e-9)
        
    # Final safety checks: ensure finite and positive
    H = np.where(np.isfinite(H), H, 1e-9)
    H = np.where(H > 0, H, 1e-9)
    
    return H
