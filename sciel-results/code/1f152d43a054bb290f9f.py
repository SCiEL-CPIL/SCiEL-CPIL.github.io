import numpy as np

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i.
    Larger values are more attractive.
    
    Strategy: Use a smooth, bounded heuristic that rewards tight fits (low waste)
    but with a relatively flat decay profile to allow the ACO to explore various
    good packing combinations rather than converging too quickly on single tightest pairs.
    """
    d = np.asarray(demands, dtype=float)
    n = len(d)
    
    if n == 0:
        return np.empty((0, 0), dtype=float)

    cap = float(capacity)
    
    # Pairwise sums
    pair_sum = d[:, None] + d[None, :]
    
    # Check which pairs fit
    fits = pair_sum <= cap
    
    # Calculate waste (remaining space)
    waste = cap - pair_sum
    
    # Initialize with a very low value for non-fitting pairs
    h = np.full((n, n), 1e-9, dtype=float)
    
    # For fitting pairs, apply the heuristic
    # Based on observations, a flatter decay (larger scale) performed better.
    # Inspiration 3 used scale=12. Let's try scale=15 to be slightly flatter.
    scale = 15.0
    
    # Calculate attractiveness for fitting pairs
    # h = 1 / (1 + (waste / scale)^2)
    # This ensures values are in (0, 1].
    # waste=0 -> 1.0
    # waste=15 -> 0.5
    # waste=30 -> 0.2
    # waste=60 -> 0.04
    
    # We only compute for the 'fits' mask to avoid division issues or unnecessary work,
    # though vectorized ops are fast.
    # Using np.where or masking. Masking is safer for potential edge cases if any.
    
    # To avoid computing on non-fitting parts which might have negative waste (though masked),
    # we can just compute on the whole array and then mask.
    # However, waste can be negative for non-fitting pairs.
    # (negative / scale)^2 is positive, so 1/(1+pos) is valid.
    # But we want to set non-fitting to 1e-9 anyway.
    
    # Compute for all, then mask
    denom = 1.0 + (waste / scale) ** 2
    h_all = 1.0 / denom
    
    # Apply mask: if it doesn't fit, set to 1e-9
    h = np.where(fits, h_all, 1e-9)
    
    # Safety checks: ensure no non-finite values or non-positive values
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9
    
    return h
