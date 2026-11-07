"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute one attractiveness score per item for ACO of multi-constraint knapsack.
    
    prizes: (5, 100) or (100,) uniform [0,1]
    weights: (5, 100) uniform [0,1]
    Returns: (100,) array, larger = more attractive.
    
    Strategy: Competitive exclusion score.
    For each item, estimate the opportunity cost of selecting it by measuring
    how many other items would become infeasible (violate at least one constraint)
    if this item were added to a near-full knapsack. Items that "exclude" fewer
    competitors have higher marginal utility.
    
    Additionally blend in per-constraint efficiency to handle ties.
    """
    start_time = time.time()
    deadline = 320.0  # Safety margin before 330s kill
    
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle shape of prizes
    if prizes.ndim == 2:
        item_prizes = np.sum(prizes, axis=0)
    else:
        item_prizes = prizes
    
    # Ensure weights is 2D
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
        
    n_constraints, n_items = weights.shape
    
    # Capacity per constraint
    capacity = 25.0
    
    # Compute per-constraint efficiency: prize / weight_c
    safe_weights = np.where(weights > 1e-9, weights, 1e-9)
    eff = item_prizes / safe_weights  # (5, 100)
    
    # Baseline: harmonic mean efficiency
    inv_eff = 1.0 / eff
    harm_eff = n_constraints / np.sum(inv_eff, axis=0)
    
    # Competitive exclusion score:
    # For item i, count how many other items j would violate at least one constraint
    # if both i and j were in a knapsack that is already at capacity in the tightest constraint.
    # Approximation: for each constraint c, the "pressure" of item i is weights[c,i].
    # An item j competes with i on constraint c if weights[c,j] > (capacity - weights[c,i]) / some_threshold
    # Simpler proxy: count items that have high weight on the same constraints where i is heavy.
    
    # Vectorized competitive exclusion:
    # For each pair (i, j), check if adding i would make j infeasible in any constraint.
    # This is O(n^2 * constraints) = 100^2 * 5 = 50000 ops, very fast.
    
    # For item i, the remaining capacity in constraint c after adding i is:
    # rem[c, i] = capacity - weights[c, i]
    rem = capacity - weights  # (5, 100)
    
    # Item j is excluded by item i in constraint c if weights[c, j] > rem[c, i]
    # i.e., weights[c, j] + weights[c, i] > capacity
    # We want to count, for each i, how many j are excluded in AT LEAST ONE constraint.
    
    # Build exclusion matrix: excl[i, j] = True if item i excludes item j in any constraint
    # excl[i, j] = any_c( weights[c, i] + weights[c, j] > capacity )
    
    # weights: (5, 100)
    # sum_pair[c, i, j] = weights[c, i] + weights[c, j]
    # Use broadcasting: weights[:, :, None] + weights[:, None, :] -> (5, 100, 100)
    sum_pair = weights[:, :, None] + weights[:, None, :]  # (5, 100, 100)
    
    # excluded_in_c[i, j] = sum_pair[c, i, j] > capacity
    excluded_any = np.any(sum_pair > capacity, axis=0)  # (100, 100)
    
    # For item i, count how many other items j are excluded by i
    # excl_count[i] = sum_j excluded_any[i, j] - 1 (exclude self)
    excl_count = np.sum(excluded_any, axis=1) - 1.0  # (100,)
    
    # Competitive exclusion score: fewer exclusions = better
    # Use inverse: score = 1 / (excl_count + 1)
    comp_score = 1.0 / (excl_count + 1.0)
    
    # Blend with efficiency
    # Normalize both to mean 1
    comp_mean = np.mean(comp_score)
    if comp_mean > 0:
        comp_score = comp_score / comp_mean
    
    harm_mean = np.mean(harm_eff)
    if harm_mean > 0:
        harm_eff = harm_eff / harm_mean
    
    # Weighted blend: emphasize competitive exclusion more
    h = 0.6 * comp_score + 0.4 * harm_eff
    
    # Ensure positive
    h = np.where(h > 0, h, 1e-9)
    
    # Normalize to mean 1
    mean_h = np.mean(h)
    if mean_h > 0:
        h = h / mean_h
    
    # Deadline guard
    if time.time() > deadline:
        # Return current result
        return h
    
    return h
# EVOLVE-BLOCK-END
