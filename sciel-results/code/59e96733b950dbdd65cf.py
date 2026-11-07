"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # 1. Calculate pair sums and feasibility
    pair_sum = demands[:, None] + demands[None, :]
    fits = pair_sum <= capacity
    
    # 2. Calculate scarcity for each item j
    # partners_j is the number of items i such that demands[i] + demands[j] <= capacity
    # Note: The ACO places item j after item i. The feasibility is symmetric.
    # So the number of valid predecessors for j is the same as valid successors.
    partners = (demands[:, None] + demands[None, :] <= capacity).sum(axis=1)
    
    # Avoid division by zero, though with d in [20, 99] and C=150,
    # an item of size 99 needs partner <= 51.
    # An item of size 20 needs partner <= 130 (all).
    # So partners should be > 0 for all items in this range.
    # But to be safe:
    partners = np.maximum(partners, 1)
    
    # Normalize scarcity to [0, 1]
    # Higher scarcity means fewer partners (harder to place)
    min_partners = np.min(partners)
    max_partners = np.max(partners)
    
    if max_partners > min_partners:
        scarcity_scores = (max_partners - partners) / (max_partners - min_partners)
    else:
        scarcity_scores = np.ones(n)
        
    # 3. Calculate size/tightness score for each pair
    # Higher pair_sum is better (fills bin more)
    # Normalize to [0, 1] relative to capacity
    size_scores = pair_sum / capacity
    
    # 4. Combine scores
    # Based on measured improvement: 0.3 * size + 0.7 * scarcity
    # We apply the scarcity score of the destination item j to the pair (i, j)
    # scarcity_scores is a vector of length n. We need to broadcast it to the columns (j)
    heuristic_matrix = 0.3 * size_scores + 0.7 * scarcity_scores[None, :]
    
    # 5. Apply feasibility mask
    # Non-positive or non-fitting become 1e-9
    # The problem states: "Heuristic entries that are not positive become 1e-9"
    # Our scores are positive. We just need to mask out infeasible pairs.
    heuristic_matrix = np.where(fits, heuristic_matrix, 1e-9)
    
    # Ensure no NaNs or Infs
    heuristic_matrix = np.nan_to_num(heuristic_matrix, nan=1e-9, posinf=1e-9, neginf=1e-9)
    
    return heuristic_matrix
# EVOLVE-BLOCK-END
