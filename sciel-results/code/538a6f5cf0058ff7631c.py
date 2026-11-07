import numpy as np

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing algorithm.
    
    The heuristic eta[i, j] represents the attractiveness of placing item j
    immediately after item i in the same bin.
    
    Strategy:
    We use a composite heuristic that combines:
    1. Best-fit preference: smaller gap is more attractive (inverse gap)
    2. Fill-ratio signal: higher utilization of the bin is preferred
    3. Complementarity signal: items with fewer valid partners are more attractive
       to place, as they are harder to fit elsewhere (scarcity-based centrality)
    
    Formula: eta[i, j] = (w_j / (gap + eps)) * (1 + alpha * (fill_ratio)^2) * (1 + beta * scarcity_j)
    where gap = capacity - w_i - w_j, fill_ratio = (w_i + w_j) / capacity
    scarcity_j is the inverse of the number of items that can fit with item j.
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
        # Add epsilon to avoid division by zero for exact fits
        eps = 1e-9
        safe_gaps = np.where(fits, gaps + eps, 1.0)
        
        # Numerator: size of the item being placed (j)
        numerator = W[None, :]
        
        # Base best-fit term: w_j / (gap + eps)
        base_term = numerator / safe_gaps
        
        # Fill ratio: how much of the capacity is used by this pair
        # Higher fill ratio means better bin utilization
        fill_ratio = np.where(fits, pair_sums / capacity, 0.0)
        
        # Secondary signal: reward high fill ratios with a smooth quadratic
        # This helps guide ACO toward combinations that fill bins well
        # while the primary inverse-gap term maintains best-fit preference
        alpha = 0.3  # Weight for the fill-ratio bonus
        fill_bonus = 1.0 + alpha * (fill_ratio ** 2)
        
        # Complementarity signal: scarcity of valid partners for item j
        # Count how many items can fit with each item j (degree in compatibility graph)
        # For each item j, count items i such that w_i + w_j <= capacity
        degree_j = np.sum(fits, axis=0)  # Shape (n,), degree of each item j
        # Avoid division by zero; items with no partners get high scarcity
        degree_j_safe = np.where(degree_j > 0, degree_j, 1.0)
        # Scarcity: 1/degree, normalized to [0, 1] range
        # Max possible degree is n, so scarcity is in [1/n, 1]
        scarcity = 1.0 / degree_j_safe
        # Normalize scarcity to [0, 1] for consistent scaling
        scarcity_max = np.max(scarcity) if np.any(scarcity > 0) else 1.0
        scarcity_norm = scarcity / scarcity_max if scarcity_max > 0 else scarcity
        
        # Tertiary signal: reward items with fewer partners (harder to place)
        # This guides ACO to place difficult items first when they fit
        beta = 0.2  # Weight for the scarcity bonus
        scarcity_bonus = 1.0 + beta * scarcity_norm[None, :]
        
        # Combine: base best-fit multiplied by fill-ratio bonus and scarcity bonus
        H_valid = base_term * fill_bonus * scarcity_bonus
        
        # Set invalid pairs to 1e-9
        H = np.where(fits, H_valid, 1e-9)
        
    # Final safety checks: ensure finite and positive
    H = np.where(np.isfinite(H), H, 1e-9)
    H = np.where(H > 0, H, 1e-9)
    
    return H
