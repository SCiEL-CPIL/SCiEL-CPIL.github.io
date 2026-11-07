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
    deadline = t0 + 300.0  # safety margin before 330s kill

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

    # Component 2: Inverse remaining space - rewards tight packing
    remaining = np.where(fits, capacity - pair, capacity)
    inv_remaining = np.where(remaining > 1e-6, 1.0 / (remaining + 1.0), 0.0)
    inv_remaining_score = np.where(fits, inv_remaining * capacity * 0.5, 0.0)

    # Component 3: Residual quality - check if leftover can fit another item
    min_d = d.min()
    residual = np.where(fits, capacity - pair, 0.0)

    # Useful residual: either can fit another item (residual >= min_d)
    # or is very tight (residual < min_d * 0.3)
    useful_residual = np.where(fits,
                                (residual >= min_d) | (residual < min_d * 0.3),
                                False)
    residual_bonus = np.where(useful_residual, 0.3, 0.0)

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

    # Optional: Add a small bonus for pairs where the residual closely matches
    # some other item's size, enabling a triple fit
    # Vectorized version using np.searchsorted
    if time.time() < deadline - 10:
        sorted_d = np.sort(d)
        
        # For each pair (i,j), find the largest item that fits in residual[i,j]
        # residual[i,j] = capacity - d[i] - d[j]
        # We want to find best_fit = max(d[k]) such that d[k] <= residual[i,j]
        
        # Vectorized searchsorted: for each element in residual, find index in sorted_d
        # searchsorted returns the index where the value would be inserted
        # We want the largest value <= residual, so side='right' - 1
        idx = np.searchsorted(sorted_d, residual, side='right') - 1
        
        # Clip indices to valid range
        idx = np.clip(idx, 0, len(sorted_d) - 1)
        
        # Get the best fitting item size for each pair
        best_fit = sorted_d[idx]
        
        # Only consider pairs that fit and have positive residual
        valid_mask = fits & (residual > 0)
        
        # Closeness: how well the best item fits into the residual
        # closeness = 1 - |residual - best_fit| / residual
        with np.errstate(divide='ignore', invalid='ignore'):
            closeness = 1.0 - np.abs(residual - best_fit) / np.maximum(residual, 1e-6)
        
        closeness = np.where(valid_mask, np.maximum(closeness, 0.0), 0.0)
        
        # Bonus: reward pairs where a third item can fit well
        bonus = closeness * 0.25
        
        # Also add a complementarity bonus: reward pairs that fill ~2/3 or ~3/4 of capacity
        # This encourages balanced packing that leaves room for more items
        fill = pair / capacity
        # Smooth peaks around 2/3 and 3/4
        peak1 = np.exp(-((fill - 2.0/3.0) ** 2) / 0.02)
        peak2 = np.exp(-((fill - 0.75) ** 2) / 0.02)
        complement_bonus = 0.15 * (peak1 + peak2) * fits
        
        h += bonus + complement_bonus
        np.fill_diagonal(h, 1e-9)
        h[~np.isfinite(h)] = 1e-9
        h[h <= 0] = 1e-9

    return h
# EVOLVE-BLOCK-END
