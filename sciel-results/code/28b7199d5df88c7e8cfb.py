"""Rewritten program: complementarity-based heuristic for ACO bin packing."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Compute a heuristic matrix for ACO bin packing.
    
    The heuristic uses a complementarity signal: for each pair (i, j),
    the attractiveness is based on:
    1. Whether items i and j fit together in a bin
    2. The scarcity of valid partners for item j (inverse degree in compatibility graph)
    3. The waste minimization (smaller remaining space is better)
    4. A centrality measure based on how many items can follow j
    
    Larger values are more attractive. Non-positive or non-finite values are set to 1e-9.
    """
    demands = np.asarray(demands, dtype=float)
    n = demands.shape[0]
    
    # Check for deadline - if we're running out of time, return a simple heuristic
    # This is a safety guard; the actual computation is fast for n=120
    
    # Compatibility matrix: can item j fit after item i in the same bin?
    pair_sum = demands[:, None] + demands[None, :]
    fits = (pair_sum <= capacity).astype(float)
    
    # Waste after placing both items: capacity - demands[i] - demands[j]
    waste = capacity - pair_sum
    
    # For each item j, compute its degree (number of items that can precede it)
    # This is the column sum of the fits matrix (how many i's can go before j)
    degree_j = fits.sum(axis=0)  # shape (n,)
    
    # For each item i, compute its degree (number of items that can follow it)
    # This is the row sum of the fits matrix (how many j's can go after i)
    degree_i = fits.sum(axis=1)  # shape (n,)
    
    # Inverse degree: items with fewer valid partners are more "critical"
    # Use a smooth inverse to avoid division by zero
    inv_degree_j = 1.0 / (degree_j + 1.0)  # shape (n,)
    inv_degree_i = 1.0 / (degree_i + 1.0)  # shape (n,)
    
    # Build the heuristic:
    # Base signal: prefer pairs that fit and minimize waste
    # Complementarity: prefer placing item j after item i if j has few options
    # (i.e., j is "hard to place" and should be placed early)
    
    # The heuristic for pair (i, j):
    # - 0 if they don't fit
    # - Otherwise, a combination of:
    #   * Low waste (tight fit is good)
    #   * High scarcity of j (if j has few valid predecessors, placing it now is valuable)
    #   * High scarcity of i (if i has few valid successors, we should use its "slot" wisely)
    
    # Normalize waste to [0, 1] range: waste / capacity
    norm_waste = waste / capacity
    
    # The complementarity score for pair (i, j):
    # We want to encourage placing "hard-to-place" items (low degree) in positions
    # where they fit. The idea is that if item j has few valid partners, it's
    # critical to place it when possible.
    
    # heuristic[i, j] = fits[i, j] * (1 - norm_waste[i, j]) * (inv_degree_j[j] ** alpha)
    # where alpha controls how strongly we weight the scarcity
    
    # Let's use a weighted combination:
    # - Tightness: (1 - norm_waste) rewards tight fits
    # - Scarcity of j: inv_degree_j[j] rewards placing scarce items
    # - Scarcity of i: inv_degree_i[i] rewards using slots for scarce predecessors
    
    alpha = 1.5  # Exponent for scarcity weighting
    
    # Compute the raw heuristic
    tightness = 1.0 - norm_waste  # shape (n, n), higher = tighter fit
    scarcity_j = inv_degree_j[None, :] ** alpha  # shape (1, n)
    scarcity_i = inv_degree_i[:, None] ** alpha  # shape (n, 1)
    
    # Combine: the heuristic is the product of these factors
    # This creates a multiplicative interaction where all factors must be good
    raw_heuristic = fits * tightness * scarcity_j * scarcity_i
    
    # Ensure the diagonal is handled (item after itself doesn't make sense in practice,
    # but the ACO algorithm will only consider unplaced items, so this is safe)
    # However, let's set the diagonal to 0 to be safe (an item can't follow itself)
    np.fill_diagonal(raw_heuristic, 0.0)
    
    # Replace non-positive or non-finite values with 1e-9
    raw_heuristic = np.where(
        (raw_heuristic > 0) & np.isfinite(raw_heuristic),
        raw_heuristic,
        1e-9
    )
    
    # Normalize to a reasonable range to avoid numerical issues
    # The ACO algorithm uses pheromone * heuristic, so we want the heuristic
    # to be on a similar scale to pheromone (which starts at 1)
    max_val = raw_heuristic.max()
    if max_val > 0:
        raw_heuristic = raw_heuristic / max_val
    
    # Ensure all values are at least 1e-9
    raw_heuristic = np.maximum(raw_heuristic, 1e-9)
    
    return raw_heuristic
# EVOLVE-BLOCK-END
