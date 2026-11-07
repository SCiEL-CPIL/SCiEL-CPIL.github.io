"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _two_pointer_complements(demands, capacity):
    """
    Use two-pointer sweep on sorted demands to find ideal partner pairs.
    Returns an array of ideal partner indices for each item in the original order.
    """
    n = len(demands)
    sorted_demands = np.sort(demands)
    sorted_indices = np.argsort(demands)
    
    # Two-pointer sweep: for each item, find the best complement
    ideal_partner = np.full(n, -1, dtype=np.int64)
    
    # For each item in sorted order, find the largest item that fits with it
    # This gives us the "ideal" complement in terms of waste minimization
    j = n - 1
    for i in range(n):
        # Find the largest j >= i such that demands[i] + demands[j] <= capacity
        while j > i and sorted_demands[i] + sorted_demands[j] > capacity:
            j -= 1
        if j >= i:
            ideal_partner[sorted_indices[i]] = sorted_indices[j]
        else:
            ideal_partner[sorted_indices[i]] = i  # fallback: pair with self
    
    return ideal_partner


def _build_complementarity_matrix(demands, capacity):
    """
    Build a heuristic matrix based on complementarity scores.
    Rewards pairs that are close to the ideal partner structure from two-pointer sweep.
    """
    n = len(demands)
    pair = demands[:, None] + demands[None, :]
    fits = pair <= capacity
    
    # Compute waste for each pair
    waste = capacity - pair
    
    # Get ideal partner indices
    ideal_partner = _two_pointer_complements(demands, capacity)
    
    # For each item i, compute how close item j is to i's ideal partner
    # We want to reward pairs where j is close to ideal_partner[i]
    
    # Create a matrix where entry [i, j] measures how good j is as a partner for i
    # Use a combination of:
    # 1. Waste-based score (lower waste = better)
    # 2. Complementarity bonus if j is the ideal partner or close to it
    
    # Base score: inverse waste (higher for tighter fits)
    # For fitting pairs, use 1/(waste+1) to avoid division by zero
    base_score = np.where(fits, 1.0 / (np.abs(waste) + 1.0), 1e-9)
    
    # Complementarity bonus: give extra weight to pairs near the ideal partner
    # Create a distance matrix from ideal partner structure
    # For each i, the ideal partner is ideal_partner[i]
    # We want to boost entries where j is close to ideal_partner[i] in the sorted order
    
    sorted_indices = np.argsort(demands)
    sorted_positions = np.empty(n, dtype=np.int64)
    sorted_positions[sorted_indices] = np.arange(n)
    
    # For each item i, compute the sorted position of its ideal partner
    ideal_positions = sorted_positions[ideal_partner]
    
    # Create a matrix where entry [i, j] = |sorted_position[j] - ideal_position[i]|
    pos_j = sorted_positions[None, :]  # shape (1, n)
    ideal_pos_i = ideal_positions[:, None]  # shape (n, 1)
    
    # Distance in sorted order from ideal partner
    dist_from_ideal = np.abs(pos_j - ideal_pos_i)
    
    # Complementarity score: exponential decay from ideal partner
    # Pairs closer to the ideal partner get higher scores
    sigma = 3.0  # width of the decay
    complementarity = np.exp(-dist_from_ideal / sigma)
    
    # Combine: base score * complementarity boost
    # For fitting pairs, multiply by complementarity to emphasize ideal partners
    # For non-fitting pairs, keep very low score
    heuristic = np.where(fits, base_score * (0.5 + 0.5 * complementarity), 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic


def heuristics(demands, capacity):
    """
    Returns an n x n matrix of attractiveness for placing item j right after item i
    in the same bin. Larger values are more attractive.
    """
    demands = np.asarray(demands, dtype=np.float64)
    n = len(demands)
    
    # Compute the complementarity-based heuristic
    heuristic = _build_complementarity_matrix(demands, capacity)
    
    # Final safety check
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
