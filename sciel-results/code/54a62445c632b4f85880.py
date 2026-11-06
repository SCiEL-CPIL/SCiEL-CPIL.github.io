"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for adaptive parameters
_start_time = time.time()
_deadline = 150.0  # seconds

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score = better choice.
    
    Uses a hybrid First-Fit-Diminishing / Best-Fit heuristic:
    - Primary: minimize remaining space (best-fit) to consolidate items
    - Secondary: relative penalty for gaps that are too small to be useful
      but too large to be ignored (fragmentation risk)
    
    For Weibull-distributed items with capacity 100, we want to:
    1. Aggressively fill bins (best-fit) to keep other bins available
    2. Avoid creating gaps that are "wasted" (small but not zero)
    """
    remaining = bins - item
    valid = remaining >= 0
    
    if not np.any(valid):
        return np.full(len(bins), -np.inf)
    
    # Core best-fit score: negative remaining (higher = less remaining = better)
    bf_score = np.where(valid, -remaining, -np.inf)
    
    # Relative gap penalty: penalize gaps that are small relative to item size
    # This avoids creating unusable slivers while still preferring tight fits
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(valid, remaining / np.maximum(item, 1e-9), 0)
    
    # Penalty is highest when ratio is in [0.05, 0.3] range
    # These gaps are too small for typical future items but not zero
    # Use a smooth penalty that peaks around ratio=0.15
    # penalty = -alpha * exp(-(ratio - 0.15)^2 / (2 * sigma^2))
    # We want to SUBTRACT this from score (i.e., add negative penalty)
    
    sigma = 0.1
    alpha = 0.5
    penalty = np.where(valid, -alpha * np.exp(-((ratio - 0.15) ** 2) / (2 * sigma ** 2)), 0)
    
    # Combine: best-fit + penalty
    score = bf_score + penalty
    
    # Mask invalid bins
    score = np.where(valid, score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
