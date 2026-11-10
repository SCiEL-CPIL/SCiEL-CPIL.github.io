"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing problem.
    
    Args:
        demands: Array of item sizes.
        capacity: Bin capacity.
        
    Returns:
        Matrix of shape (N, N) where entry (i, j) is the attractiveness
        of placing item j after item i.
    """
    n = len(demands)
    # Compute all pairwise sums
    pair_sums = demands[:, None] + demands[None, :]
    
    # Initialize heuristic matrix with a very small value for infeasible pairs
    heur = np.full((n, n), 1e-9)
    
    # Mask for feasible pairs (sum <= capacity)
    feasible_mask = pair_sums <= capacity
    
    # For feasible pairs, calculate attractiveness.
    # We use a power law to strongly favor tight packing (high utilization).
    # Base ratio: (d_i + d_j) / capacity
    # Power: 3.0 to sharpen the preference for bins that are nearly full.
    # This helps the ACO converge to solutions with fewer bins by making
    # "loose" pairs significantly less attractive than "tight" pairs.
    
    # Avoid division by zero or negative values, though demands are positive.
    ratios = np.zeros_like(pair_sums, dtype=float)
    ratios[feasible_mask] = pair_sums[feasible_mask] / capacity
    
    # Apply power law
    heur[feasible_mask] = ratios[feasible_mask] ** 3
    
    # Ensure diagonal (placing item after itself) is handled.
    # Usually, we don't place an item after itself in the same bin unless it's a multi-item bin,
    # but the ACO logic "picks a fitting item" implies picking from remaining items.
    # The matrix is used as pheromone[i, j] * heuristic[i, j].
    # If i == j, it means placing item i after item i, which is invalid in a single-item-per-slot context 
    # unless we allow multiple copies, but here items are distinct.
    # The ACO loop likely prevents picking the current item again.
    # However, to be safe, we can leave diagonal as calculated or set to 0.
    # If the ACO prevents picking 'current' item, diagonal value doesn't matter.
    # If it doesn't, we should probably set diagonal to 0 or 1e-9 to avoid self-selection.
    # The prompt says "picks a fitting item". It doesn't explicitly exclude the current item, 
    # but "placing item j right after item i" usually implies i != j in a sequence of distinct items.
    # Let's set diagonal to 1e-9 to be safe, as you can't place an item after itself in a bin 
    # if it's already there (unless it's a different instance of the same item, but items are distinct).
    np.fill_diagonal(heur, 1e-9)
    
    return heur
# EVOLVE-BLOCK-END
