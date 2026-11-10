"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    d = np.asarray(demands, dtype=float)
    pair = d[:, None] + d[None, :]
    mask = pair <= capacity
    # For pairs that fit, we want a high attractiveness.
    # The heuristic should encourage placing items that fit together tightly.
    # A good measure is how "full" the bin becomes: (pair / capacity).
    # However, we also want to prioritize larger items or items that are harder to place.
    # Let's try a combination:
    # 1. Base attractiveness: how well they fit (pair / capacity). Higher is better (closer to 1).
    # 2. Size weight: larger items are harder to place, so placing them in a good spot is more critical.
    #    We can weight by the average size of the pair, or the max size.
    
    # Let's try: attractiveness = (pair / capacity) * (pair / capacity)
    # This squares the "fullness", making tight fits significantly more attractive.
    
    # Or, let's look at the complement: (capacity - pair) is the wasted space.
    # We want to minimize wasted space. So attractiveness ~ 1 / (1 + wasted_space).
    # But if wasted_space is 0, this is 1. If wasted_space is large, it's small.
    
    # Let's try a simple approach first:
    # Heuristic = 1 if fits, 0 if not. But that's too binary.
    # The seed used: pair / capacity if fits, else 1e-9.
    # This means a pair of 150/150 = 1.0, 75+75=150 -> 1.0.
    # 20+20=40 -> 40/150 = 0.26.
    # So it prefers tight fits.
    
    # Let's enhance this by considering the "difficulty" of the items.
    # Large items are harder to fit. If we place a large item next to another large item,
    # and they fit, that's a very good move.
    
    # Idea: Heuristic = (pair / capacity) ** alpha * (max(d_i, d_j) / max_demand) ** beta
    # Let's try alpha=2, beta=1.
    
    max_demand = np.max(d)
    if max_demand == 0:
        return np.ones((n, n))
    
    max_size_weight = np.maximum(d[:, None], d[None, :]) / max_demand
    
    # Base fit score
    fit_score = np.where(mask, pair / capacity, 1e-9)
    
    # Squared fit score to emphasize tight fits
    fit_score_sq = fit_score ** 2
    
    # Combine with size weight
    # We want to reward placing large items in tight fits.
    # heuristic = fit_score_sq * max_size_weight
    
    # Let's try a different angle:
    # The ant chooses j given i.
    # If i is large, we need j to be small enough to fit, and ideally fill the bin.
    # If i is small, j can be large.
    
    # Let's try: heuristic = (capacity - pair) is bad.
    # So attractiveness = 1 / (1 + (capacity - pair))
    # This is smooth and favors tight fits.
    
    wasted = np.where(mask, capacity - pair, capacity)
    attractiveness = 1.0 / (1.0 + wasted)
    
    # But we also want to consider the size of the items.
    # Let's multiply by the average size of the pair, normalized.
    avg_size = (d[:, None] + d[None, :]) / (2.0 * max_demand)
    
    # Combine: attractiveness * avg_size
    # This rewards tight fits of large items.
    
    result = attractiveness * avg_size
    
    # Ensure non-fitting pairs are 1e-9
    result = np.where(mask, result, 1e-9)
    
    # Ensure no zeros or negatives
    result = np.maximum(result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
