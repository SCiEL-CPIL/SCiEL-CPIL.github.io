"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_DEADLINE = None

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 115.0  # 5s safety margin
    
    remaining = bins - item
    # Only consider bins with enough remaining capacity
    valid = remaining >= 0
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid):
        return scores
    
    valid_remaining = remaining[valid]
    
    # Number of bins that are already in use (have been filled at least once)
    # bins[i] < 100 means bin i has been used (assuming capacity=100)
    # But we don't know capacity directly, so use: bins[i] < max(bins) as proxy
    # Better: count how many bins have non-zero usage
    # Since we only see remaining capacity, we can't directly know usage count.
    # Instead, use a heuristic: prefer bins that are "more filled" (less remaining)
    # but also consider consolidation: if many bins have similar remaining, prefer
    # the one with least remaining to consolidate.
    
    # Smooth best-fit with a slight preference for bins that are closer to full
    # This helps consolidate items into fewer bins
    # Use a soft-min approach: score based on how tight the fit is
    # Tighter fit = better (lower remaining after placement)
    
    # Primary: minimize remaining space (best-fit)
    # Secondary: prefer bins that are already "warmed up" (have been used)
    # We can approximate "used" by checking if remaining < some threshold
    # But without knowing capacity, use relative measure
    
    # Simple improvement: weight the best-fit score by considering
    # the distribution of remaining capacities to encourage consolidation
    n_valid = len(valid_remaining)
    if n_valid > 1:
        # Normalize remaining to [0, 1] for the valid bins
        r_min = np.min(valid_remaining)
        r_max = np.max(valid_remaining)
        r_range = r_max - r_min
        if r_range > 1e-9:
            norm_remaining = (valid_remaining - r_min) / r_range
        else:
            norm_remaining = np.zeros(n_valid)
        
        # Score: strongly prefer tight fits, with a small bonus for
        # bins that are in the "middle" of usage (not nearly empty, not full)
        # This encourages consolidation without over-packing
        scores_valid = -valid_remaining * 1.0  # primary: best-fit
        
        # Add a small consolidation incentive: prefer bins where remaining
        # is not too large (avoid creating many partially-filled bins)
        # This is already captured by -remaining, but let's add a
        # quadratic term to strongly penalize loose fits
        scores_valid = scores_valid - 0.1 * valid_remaining**2
        
        # Small tie-breaker: among similar fits, prefer the bin with
        # more remaining capacity (more flexible for future items)
        # Actually no - prefer tighter fit. The -0.1*remaining^2 already does this.
        
        scores[valid] = scores_valid
    else:
        scores[valid] = -valid_remaining
    
    return scores
# EVOLVE-BLOCK-END
