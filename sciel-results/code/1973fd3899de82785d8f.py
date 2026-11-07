"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Compute an item-pair attractiveness matrix for ant colony optimization
    of offline bin packing.
    
    The heuristic guides ants to place item j after item i in a way that
    minimizes wasted space and leaves useful residuals for future items.
    """
    t0 = time.time()
    deadline = t0 + 310.0  # safety margin before 330s kill

    n = len(demands)
    d = np.array(demands, dtype=float)

    # Pair sums
    pair = d[:, None] + d[None, :]

    # Mask for pairs that fit
    fits = pair <= capacity

    # Initialize with very low attractiveness
    h = np.full((n, n), 1e-9, dtype=float)

    # Component 1: Fill ratio squared - rewards fuller packs quadratically
    fill_ratio = pair / capacity
    fill_score = np.where(fits, fill_ratio ** 2, 0.0)

    # Component 2: Inverse remaining space - rewards tight packing (sharpened)
    remaining = np.where(fits, capacity - pair, capacity)
    # Use a steeper curve: 1/(remaining+1)^2 amplifies the difference between
    # tight (remaining~5) and loose (remaining~50) packs
    inv_remaining = np.where(remaining > 1e-6, 1.0 / (remaining + 1.0) ** 2, 0.0)
    inv_remaining_score = np.where(fits, inv_remaining * capacity * capacity * 0.01, 0.0)

    # Component 3: Smooth residual quality - continuous signal replacing the
    # hard threshold at min_d. High when residual is very small (tight) or
    # close to a "useful" size (can fit another item).
    min_d = d.min()
    residual = np.where(fits, capacity - pair, 0.0)

    # Smooth bonus: peaks at residual=0 (perfect fill) and has a secondary
    # peak around residual=min_d (can fit one more smallest item)
    # Use a sum of two smooth bumps
    sigma_tight = min_d * 0.4 + 1.0
    tight_bump = np.exp(-residual ** 2 / (2 * sigma_tight ** 2))
    sigma_useful = min_d * 0.5 + 1.0
    useful_bump = np.exp(-(residual - min_d) ** 2 / (2 * sigma_useful ** 2))
    residual_bonus = np.where(fits, 0.35 * tight_bump + 0.15 * useful_bump, 0.0)

    # Combine components
    h = np.where(fits, fill_score + inv_remaining_score + residual_bonus, 1e-9)

    # Don't allow placing an item after itself
    np.fill_diagonal(h, 1e-9)

    # Ensure non-positive and non-finite values become 1e-9
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9

    # Check deadline before optional enhancement
    if time.time() > deadline - 5:
        return h

    # Vectorized triple-fit bonus: for each fitting pair (i,j), find the
    # largest item that fits into the residual and reward closeness
    sorted_d = np.sort(d)
    
    # For each pair (i,j), residual = capacity - d[i] - d[j]
    # Find the largest item <= residual using searchsorted
    # residual is an (n,n) array; we need to vectorize searchsorted
    # searchsorted works on 1D arrays, so we flatten, search, then reshape
    
    r_flat = residual.ravel()
    # Only process fitting pairs
    fits_flat = fits.ravel()
    
    # For non-fitting pairs, residual is 0, searchsorted will give idx=-1
    # Clip to valid range
    idx = np.searchsorted(sorted_d, r_flat, side='right') - 1
    idx = np.clip(idx, 0, len(sorted_d) - 1)
    
    # Best fit item size for each pair
    best_fit_flat = sorted_d[idx]
    best_fit = best_fit_flat.reshape(n, n)
    
    # Tighter triple-fit bonus: reward only when the residual is well-utilized
    # by the next item. Use a sharper penalty for poor fit.
    # closeness^2 amplifies near-perfect fits vs mediocre ones
    r_safe = np.where(r_flat.reshape(n, n) > 1e-6, r_flat.reshape(n, n), 1.0)
    closeness = 1.0 - np.abs(residual - best_fit) / r_safe
    closeness = np.maximum(0.0, closeness)
    closeness_sq = closeness ** 2  # sharper: 0.5 -> 0.25, 0.9 -> 0.81
    
    # Only apply where fits and residual > 0 and i != j
    triple_mask = fits & (residual > 0) & ~np.eye(n, dtype=bool)
    triple_bonus = np.where(triple_mask, closeness_sq * 0.25, 0.0)
    
    h += triple_bonus
    
    # Component 4: Size balance bonus - reward pairs where both items are
    # reasonably sized (not one tiny + one huge). This promotes balanced bins.
    # Balance score = 1 - |d[i] - d[j]| / max(d[i], d[j])
    # High when items are similar size, low when very different
    max_d_pair = np.maximum(d[:, None], d[None, :])
    diff = np.abs(d[:, None] - d[None, :])
    balance = 1.0 - diff / np.maximum(max_d_pair, 1e-6)
    balance_bonus = np.where(fits & ~np.eye(n, dtype=bool), balance * 0.15, 0.0)
    
    h += balance_bonus
    
    # Component 5: Multi-target residual bonus - reward residuals close to
    # common item sizes (modes of the size distribution). This encourages
    # packing toward sizes that actually appear in the dataset.
    # Use quantiles to find representative target sizes
    targets = np.array([np.percentile(d, p) for p in [25, 50, 75]])
    target_bonus = np.zeros((n, n), dtype=float)
    for t_size in targets:
        residual_dist = np.abs(residual - t_size)
        sigma = t_size * 0.25 + 1.0
        bump = np.exp(-residual_dist ** 2 / (2 * sigma ** 2))
        target_bonus += bump
    target_bonus *= 0.08
    target_bonus = np.where(fits & ~np.eye(n, dtype=bool) & (residual > 0), target_bonus, 0.0)
    
    h += target_bonus

    # Component 6: Dead gap penalty - penalize pairs that create residual space
    # too small for any remaining item, as these are "dead" wasted space.
    # Count how many items fit into the residual; if count is 0, it's a dead gap.
    # We reward pairs where the residual either fits many items (useful) or is very small (tight).
    # This is the inverse of the strategy: penalize dead gaps explicitly.
    
    # For each residual r, count items <= r
    # This is essentially the idx + 1 from searchsorted (number of items <= r)
    # idx was already computed above as searchsorted - 1
    # Number of items fitting = idx + 1, but clipped to [0, n-1]
    # Actually, idx is the index of the largest item <= r, so count = idx + 1 if idx >= 0
    # But we need to be careful: if r < min_d, then idx = -1, count = 0
    
    # Recompute more carefully for the dead gap logic
    idx_for_count = np.searchsorted(sorted_d, r_flat, side='right') - 1
    # Number of items that fit into residual r
    count_fitting = np.where(idx_for_count >= 0, idx_for_count + 1, 0)
    count_fitting = count_fitting.reshape(n, n)
    
    # Dead gap: count_fitting == 0 and residual > 0 (not a perfect fill)
    # Perfect fill: residual == 0, which is good (tight_bump handles this)
    # Dead gap: residual > 0 but no item fits, so this space is wasted
    dead_gap_mask = fits & (residual > 0) & (count_fitting == 0) & ~np.eye(n, dtype=bool)
    
    # Penalty for dead gaps: the larger the dead gap, the worse it is
    # But we want to be smooth. Use the residual size as the penalty weight.
    # Normalize by capacity to keep it in a reasonable range
    dead_gap_penalty = np.where(dead_gap_mask, residual / capacity * 0.3, 0.0)
    
    h -= dead_gap_penalty
    
    # Ensure values don't go negative or non-finite
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9

    # Final cleanup
    np.fill_diagonal(h, 1e-9)
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9

    return h
# EVOLVE-BLOCK-END
