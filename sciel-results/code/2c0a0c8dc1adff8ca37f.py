"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    demands = np.array(demands)
    n = len(demands)
    
    # 1. Compute compatibility matrix (fits)
    # fits[i, j] is True if item i and item j can be in the same bin
    pair_sums = demands[:, None] + demands[None, :]
    fits = pair_sums <= capacity
    
    # 2. Compute degree of each item in the compatibility graph
    # deg[j] is the number of items i (i != j) that fit with j
    # Sum along axis 0 gives count for each j
    # Subtract diagonal because an item doesn't fit with itself in this context (i != j)
    diag = np.diag(fits).astype(int)
    deg = np.sum(fits, axis=0) - diag
    
    # 3. Compute scarcity signal
    # Higher scarcity means fewer partners, so the item is "harder" to place
    # We use 1/(deg+1) to avoid division by zero and to normalize
    scarcity = 1.0 / (deg + 1)
    
    # 4. Compute efficiency signal
    # Higher efficiency means the pair fills the bin more completely
    # Normalize by capacity to keep values in a reasonable range (0, 1]
    efficiency = pair_sums / capacity
    
    # 5. Combine signals
    # The heuristic h[i, j] represents the attractiveness of placing j after i.
    # We weight the efficiency by the scarcity of j.
    # This encourages the ant to place items that have fewer options (harder items)
    # into bins where they fit, prioritizing them over "easy" items.
    # scarcity[None, :] broadcasts the scarcity of j across all i.
    H = efficiency * scarcity[None, :]
    
    # 6. Apply mask
    # Only allow transitions if they fit and i != j
    # np.eye(n) has 1s on diagonal
    valid = fits & (~np.eye(n, dtype=bool))
    H = np.where(valid, H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
