"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Compute an item-by-item attractiveness matrix for placing item j right after item i.
    Larger values are more attractive.
    """
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Pair sums: pair[i, j] = demands[i] + demands[j]
    pair = demands[:, None] + demands[None, :]
    
    # Fit mask: can item j follow item i in the same bin?
    fits = pair <= capacity
    
    # Waste after placing i then j: capacity - (d[i] + d[j])
    waste = capacity - pair
    
    # For each item j, count how many items k can be placed in the same bin as j
    # (i.e., d[j] + d[k] <= capacity). This measures "scarcity" of valid partners.
    # Items with fewer valid partners are more "scarce" and should be prioritized.
    compat_count = (demands[None, :] + demands[:, None] <= capacity).sum(axis=1)
    # compat_count[j] = number of items k such that d[j] + d[k] <= capacity
    # Lower compat_count means item j is harder to place -> more scarce
    # Use inverse as a weight: items with fewer partners get higher heuristic boost
    scarcity = 1.0 / (compat_count + 1.0)  # shape (n,)
    
    # For each pair (i, j) that fits:
    # - Base attractiveness: how well the pair fills the bin (higher fill = better)
    # - Scarcity boost: if item j is scarce, boost its placement
    # - Forward flexibility: after placing j, how much room is left for subsequent items?
    #   remaining_after_j = capacity - d[i] - d[j] = waste[i,j]
    #   We want to leave room that can accommodate more items.
    #   A simple proxy: the number of items k that fit in the remaining space.
    
    # Precompute: for each possible remaining space r, how many items fit?
    # remaining space after i,j is waste[i,j]. An item k fits if d[k] <= waste[i,j].
    # We can precompute sorted demands and use searchsorted.
    sorted_demands = np.sort(demands)
    
    # For each pair (i,j), count items k with d[k] <= waste[i,j]
    # This is the number of items that could potentially be added after j.
    # Use searchsorted on sorted_demands.
    # waste has shape (n, n), so we can vectorize.
    # num_future[i, j] = number of items k with d[k] <= waste[i, j]
    # Use np.searchsorted(sorted_demands, waste, side='right')
    num_future = np.searchsorted(sorted_demands, waste, side='right')  # shape (n, n)
    
    # Heuristic components for fitting pairs:
    # 1. Fill score: pair / capacity, raised to a power to sharpen tight fits
    fill_score = np.where(fits, pair / capacity, 0.0)
    fill_sharp = fill_score ** 1.5  # Sharper discrimination for tight fits
    
    # 2. Scarcity of item j: scarcity[j] applied to column j
    scarcity_boost = scarcity[None, :]  # shape (1, n), broadcast to (n, n)
    
    # 3. Forward flexibility: more future items that fit is better
    # Normalize num_future to [0, 1]
    max_future = num_future.max() if num_future.max() > 0 else 1
    flexibility = num_future / max_future  # shape (n, n)
    
    # Combine: heuristic[i, j] = fill_sharp[i,j] * (1 + scarcity_boost[j]) * (1 + flexibility[i,j])
    # For non-fitting pairs, use 1e-9
    heuristic = fill_sharp * (1.0 + scarcity_boost) * (1.0 + flexibility)
    heuristic = np.where(fits, heuristic, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
