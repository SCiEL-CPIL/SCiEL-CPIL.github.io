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
    partners = fits.sum(axis=1)
    partners = np.maximum(partners, 1)
    
    # 3. Calculate size/tightness score for each pair
    # Higher pair_sum is better (fills bin more)
    # Normalize to [0, 1] relative to capacity
    size_scores = pair_sum / capacity
    
    # 4. Combine scores
    # Use a more aggressive weighting for scarcity to prioritize hard-to-place items
    # Also consider the gap: smaller gap (closer to capacity) is better
    gap = capacity - pair_sum
    # Inverse gap score: higher when gap is small
    # Avoid division by zero where gap is 0 (perfect fit)
    # Use 1 / (gap + epsilon) but scale it
    eps = 1e-6
    gap_scores = 1.0 / (gap + eps)
    # Normalize gap scores to [0, 1] range roughly
    max_gap_score = np.max(gap_scores)
    if max_gap_score > 0:
        gap_scores = gap_scores / max_gap_score
    
    # Combine: weight scarcity heavily, then fit quality
    # scarcity_scores[None, :] applies to destination item j
    min_partners = np.min(partners)
    max_partners = np.max(partners)
    if max_partners > min_partners:
        scarcity_scores = (max_partners - partners) / (max_partners - min_partners)
    else:
        scarcity_scores = np.ones(n)
        
    heuristic_matrix = 0.4 * size_scores + 0.4 * scarcity_scores[None, :] + 0.2 * gap_scores
    
    # 5. Apply feasibility mask
    heuristic_matrix = np.where(fits, heuristic_matrix, 1e-9)
    
    # Ensure no NaNs or Infs
    heuristic_matrix = np.nan_to_num(heuristic_matrix, nan=1e-9, posinf=1e-9, neginf=1e-9)
    
    return heuristic_matrix
# EVOLVE-BLOCK-END
