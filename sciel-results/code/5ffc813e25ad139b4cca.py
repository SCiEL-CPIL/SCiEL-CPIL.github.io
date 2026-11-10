import numpy as np
import time

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i.
    Larger values are more attractive.
    
    Strategy: Combine the smooth bounded structure of the current program with the
    sharper decay insights from the inspirations. We use a normalized waste metric
    and a sharper power (cubic) to strongly prioritize tight fits, while keeping
    the function smooth and bounded to prevent premature convergence or numerical
    issues in the ACO probability calculation.
    """
    start_time = time.time()
    
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
    
    # Normalize waste by capacity to make the scale parameter dimensionless
    # waste_norm is in [0, 1] for fitting pairs
    waste_norm = waste / cap
    
    # Initialize with a very low value for non-fitting pairs
    h = np.full((n, n), 1e-9, dtype=float)
    
    # For fitting pairs, apply the heuristic
    # We want a sharper decay than the current program (scale=15, power=2).
    # Inspiration 1 used power=2 on unnormalized waste.
    # Knowledge suggests "Sharper decay (scale 3, power 3 ...)" improved score.
    # Let's use a normalized waste with a small scale and power 3.
    # h = 1 / (1 + (waste_norm / scale)^power)
    # If scale=0.1 and power=3:
    # waste=0 -> 1.0
    # waste=15 (0.1*cap) -> 1/(1+1) = 0.5
    # waste=30 (0.2*cap) -> 1/(1+8) = 0.11
    # waste=75 (0.5*cap) -> 1/(1+125) = 0.008
    # This provides a strong preference for tight fits while remaining bounded.
    
    scale = 0.1
    power = 3.0
    
    # Compute for all, then mask
    # We use np.where to avoid division by zero or negative base issues if any
    # although waste is non-negative for fits.
    term = (waste_norm / scale) ** power
    h_all = 1.0 / (1.0 + term)
    
    # Apply mask: if it doesn't fit, set to 1e-9
    h = np.where(fits, h_all, 1e-9)
    
    # Safety checks: ensure no non-finite values or non-positive values
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9
    
    # Deadline guard (though this computation is very fast, it's good practice)
    elapsed = time.time() - start_time
    if elapsed > 290:
        # Return the best result found so far (which is h)
        pass
        
    return h
