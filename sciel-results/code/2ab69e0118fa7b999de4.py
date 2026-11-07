"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    n = demands.shape[0]
    
    # Pair sum matrix: pair[i, j] = demands[i] + demands[j]
    pair = demands[:, None] + demands[None, :]
    
    # Fits mask: True if item j can follow item i in the same bin
    fits = pair <= capacity
    
    # Waste: remaining space after placing i then j (lower waste = tighter fit = better)
    waste = capacity - pair
    
    # For fitting pairs, we want to prefer pairs with less waste (tighter fit)
    # This encourages filling bins more completely
    # Use 1/(waste + 1) to make tighter fits more attractive
    # For non-fitting pairs, set to 1e-9
    
    # Also consider: items that are harder to place (fewer compatible partners) should be
    # given higher priority. We can encode this by boosting the heuristic for pairs
    # where the next item j has fewer compatible partners.
    
    # Count compatible partners for each item j
    compat_count = (demands[None, :] + demands[:, None] <= capacity).sum(axis=1)
    
    # Scarcity: items with fewer compatible partners are harder to place
    # Boost heuristic for placing scarce items
    # scarcity[j] is higher when compat_count[j] is lower
    scarcity = 1.0 / (compat_count + 1.0)
    
    # Build the heuristic matrix
    # For fitting pairs: attractiveness = (1/(waste+1)) * scarcity[j]
    # This encourages tight packing of hard-to-place items
    # For non-fitting pairs: 1e-9
    
    heuristic = np.full((n, n), 1e-9, dtype=float)
    
    # For fitting pairs
    inv_waste = np.zeros((n, n), dtype=float)
    inv_waste[fits] = 1.0 / (waste[fits] + 1.0)
    
    # Multiply by scarcity of the next item j
    heuristic = inv_waste * scarcity[None, :]
    
    # Ensure non-fitting pairs remain at 1e-9
    heuristic[~fits] = 1e-9
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
