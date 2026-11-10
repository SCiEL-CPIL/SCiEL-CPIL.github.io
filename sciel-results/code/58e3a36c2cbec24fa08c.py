"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ant colony optimization.
    
    Strategy: Combine the structural idea of favoring large items (product)
    with a refinement that strongly penalizes loose fits (squared inverse waste).
    Additionally, use maximum weight matching to identify globally optimal
    disjoint pairs and boost them in the heuristic to guide ACO ants toward
    proven good pairings.
    
    The heuristic encourages:
    1. Pairing items that fit together.
    2. Among fitting pairs, preferring those with larger product (large items).
    3. Among those, strongly preferring those that fill the bin more completely.
    4. Boosting pairs that are part of a globally optimal matching.
    """
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s kill
    
    # Convert to numpy array for vectorized operations
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    
    if n == 0:
        return np.empty((0, 0))
        
    # Calculate sums to check if items fit together
    sums = d[:, None] + d[None, :]
    
    # Calculate products to measure "compatibility" (favoring large-large pairs)
    products = d[:, None] * d[None, :]
    
    # Calculate waste (how much space is left in the bin after placing both items)
    waste = capacity - sums
    
    # Condition: items must fit together
    fits = sums <= capacity
    
    # Use a small epsilon to avoid division by zero when waste is exactly 0
    epsilon = 1e-6
    waste_safe = np.where(fits, waste + epsilon, 1.0)
    
    # Base heuristic: product * (1 / waste_safe^2)
    # Stronger inverse-gap penalty: heavily favors pairs that fill the bin tightly
    H_fitting = products / (waste_safe ** 2)
    H_base = np.where(fits, H_fitting, 1e-9)
    
    # Maximum weight matching approach:
    # Find globally optimal disjoint pairs using the Hungarian algorithm
    # to identify the best set of non-overlapping pairs.
    
    if n >= 2:
        # Build weight matrix for matching
        # Use the base heuristic as the weight for the matching problem
        # For non-fitting pairs, use a very small weight
        W_match = H_base.copy()
        
        # Set diagonal to 0 (no self-pairs)
        np.fill_diagonal(W_match, 0)
        
        # Ensure non-negative weights for Hungarian algorithm
        # The Hungarian algorithm minimizes cost, so we negate the weights
        # to maximize the total weight of selected pairs
        cost_matrix = -W_match
        
        # Find optimal assignment (maximum weight matching)
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Create a boolean mask for the matched pairs
            matched = np.zeros((n, n), dtype=bool)
            for i, j in zip(row_ind, col_ind):
                if i != j and fits[i, j]:
                    matched[i, j] = True
                    matched[j, i] = True  # Symmetric
            
            # Boost matched pairs by a factor
            # This provides strong guidance to ACO ants toward proven good pairings
            boost_factor = 10.0
            H_boosted = H_base * (1.0 + boost_factor * matched)
            
        except Exception:
            # If matching fails, fall back to base heuristic
            H_boosted = H_base
    else:
        H_boosted = H_base
    
    # Ensure non-fitting pairs remain at 1e-9
    H = np.where(fits, H_boosted, 1e-9)
    
    # Ensure shape is correct
    if H.shape != (n, n):
        raise ValueError("Wrong shape")
        
    # Check for non-finite values and sanitize if needed
    if not np.all(np.isfinite(H)):
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    # Ensure all values are positive
    H = np.where(H > 0, H, 1e-9)
    
    # Time guard
    elapsed = time.time() - start_time
    if elapsed > deadline:
        pass  # We're already returning, so this is just a safety check
    
    return H
# EVOLVE-BLOCK-END
