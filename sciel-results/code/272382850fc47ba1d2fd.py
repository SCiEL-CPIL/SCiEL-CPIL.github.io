"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Compute an item-pair attractiveness matrix for ant colony optimization
    of offline bin packing.
    
    The heuristic combines immediate fill efficiency with an estimate of
    future packing potential in the remaining residual space.
    """
    t0 = time.time()
    deadline = t0 + 300.0  # safety margin before 330s kill

    n = len(demands)
    d = np.array(demands, dtype=float)
    
    # 1. Compute pair sums and residuals
    d_col = d[:, None]
    d_row = d[None, :]
    pair_sum = d_col + d_row
    residual = capacity - pair_sum

    # Mask for valid pairs (residual >= 0)
    fits = residual >= 0

    # Initialize heuristic matrix
    h = np.full((n, n), 1e-9, dtype=float)

    if not np.any(fits):
        return h

    # 2. Pre-compute sorted demands and prefix sums for future potential estimation
    # Sort demands
    sorted_d = np.sort(d)
    
    # Compute prefix sums of sorted demands
    # prefix_sum[i] = sum of sorted_d[0..i-1]
    # We want sum of all d_k such that d_k <= r
    # Using searchsorted to find the count, then prefix sum to get the total weight
    prefix_sum = np.zeros(n + 1, dtype=float)
    prefix_sum[1:] = np.cumsum(sorted_d)

    # 3. For each valid pair, estimate the "Future Potential"
    # Future Potential = Total weight of items that could fit in the residual
    # This is an upper bound on how much more we can pack in this bin.
    
    # We only need to compute this for valid pairs to save time, 
    # but vectorized searchsorted is fast enough for n=120.
    
    # Find indices for each residual in sorted_d
    # side='right' gives the index where the value would be inserted to keep order
    # i.e., the number of elements <= residual
    idx = np.searchsorted(sorted_d, residual, side='right')
    
    # Get the sum of items that fit in the residual
    # prefix_sum[idx] gives the sum of the first idx items in sorted_d
    future_weight = prefix_sum[idx]
    
    # Normalize future weight by capacity to make it comparable to immediate fill
    # Scale factor: We want to reward future potential, but not dominate immediate fill.
    # If immediate fill is ~100, and future potential is ~100, total ~200.
    # Let's use a weight lambda = 0.8 for future potential.
    lambda_future = 0.8
    
    # Immediate fill score
    immediate_score = np.where(fits, pair_sum, 0.0)
    
    # Future potential score
    # Only valid for fits
    future_score = np.where(fits, lambda_future * (future_weight / capacity), 0.0)
    
    # Combine
    h = immediate_score + future_score
    
    # 4. Add a bonus for "Perfect Fits" or "Very Tight Fits"
    # If residual is very close to 0, it's an exact fill, which is ideal.
    # We add a small bonus to break ties and encourage exactness.
    tight_bonus = np.where(fits & (residual < 1.0), 5.0, 0.0)
    h += tight_bonus
    
    # 5. Cleanup
    # Ensure invalid pairs remain 1e-9
    h[~fits] = 1e-9
    
    # Don't allow placing an item after itself
    np.fill_diagonal(h, 1e-9)
    
    # Ensure non-positive and non-finite values become 1e-9
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9
    
    # Check deadline
    if time.time() > deadline - 5:
        return h
        
    # Optional: Refine with a "Complementarity" check
    # The current heuristic is a strong proxy. 
    # We can add a small term that rewards residuals that match the *median* item size closely,
    # as this suggests the next item (likely near median) will fit well.
    med_d = np.median(d)
    # Smooth peak at med_d
    peak_width = med_d * 0.2
    complementarity = np.exp(-((residual - med_d) ** 2) / peak_width)
    # Only apply if fits
    complementarity = np.where(fits, complementarity, 0.0)
    
    # Add small weight to complementarity
    h += 0.1 * complementarity
    
    np.fill_diagonal(h, 1e-9)
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9

    return h
# EVOLVE-BLOCK-END
