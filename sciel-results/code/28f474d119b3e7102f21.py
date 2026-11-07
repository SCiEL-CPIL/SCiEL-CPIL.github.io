import numpy as np

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing algorithm.
    
    The heuristic eta[i, j] represents the attractiveness of placing item j
    immediately after item i in the same bin.
    
    Strategy:
    Strong Best-Fit heuristic with scarcity weighting.
    1. Tight fit: Exponential decay with gap size (penalize waste).
    2. Item size: Prefer larger items (harder to place).
    3. Scarcity: Prefer items with fewer compatible partners (rare items).
    
    Formula: eta[i, j] = w_j * exp(-alpha * gap) * scarcity_j
    where gap = capacity - w_i - w_j.
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
        # Parameters tuned for strong best-fit preference
        # Higher alpha makes the heuristic more sensitive to gaps
        alpha = 0.15
        
        # Exponential decay of attractiveness with gap size
        # This creates a strong preference for tight fits
        # Use np.where to avoid warnings on invalid pairs, though we mask them later
        gap_penalty = np.exp(-alpha * np.maximum(gaps, 0))
        
        # Item size factor: prefer placing larger items
        # W[None, :] broadcasts to (n, n) for column j
        item_size = W[None, :]
        
        # Complementarity signal: for each item j, count how many items i can fit with j
        # Degree of j = number of i such that fits[i, j] is True
        degrees = np.sum(fits, axis=0)  # shape (n,)
        
        # Scarcity: inverse of degree, items with fewer compatible partners are more "scarce"
        # Add small epsilon to avoid division by zero
        epsilon = 1e-6
        scarcity = 1.0 / (degrees + epsilon)
        
        # Normalize scarcity to be centered around 1 for stability
        scarcity_mean = np.mean(scarcity)
        scarcity_normalized = scarcity / scarcity_mean if scarcity_mean > 0 else np.ones(n)
        
        # Composite: reward tight fits, large items, and scarce items
        # The exponential term dominates the gap sensitivity
        composite = item_size * gap_penalty * scarcity_normalized[None, :]
        
        # Set invalid pairs to 1e-9
        H = np.where(fits, composite, 1e-9)
        
    # Final safety checks: ensure finite and positive
    H = np.where(np.isfinite(H), H, 1e-9)
    H = np.where(H > 0, H, 1e-9)
    
    return H
