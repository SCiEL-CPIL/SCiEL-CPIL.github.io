"""Online bin packing: structural redesign using composite heuristic with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 120.0
_START_TIME = time.time()

def _elapsed() -> float:
    return time.time() - _START_TIME

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: A composite heuristic that balances:
    1. Best-fit (least remaining space) to minimize fragmentation
    2. A mild preference for bins that are already "warmed up" (medium fill)
       to consolidate items and leave empty bins for larger future items
    3. A penalty for bins that would become very full (risk of future large items)
    
    This is a structural departure from pure best-fit by incorporating
    a fill-level modulation term.
    """
    # Safety margin: if we're close to the deadline, just return best-fit quickly
    if _elapsed() > _DEADLINE - 5.0:
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores
    
    remaining = bins - item
    valid = remaining >= 0
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
    
    # For invalid bins, assign a very low score
    scores = np.full_like(bins, -np.inf)
    
    if not np.any(valid):
        return scores
    
    valid_remaining = remaining[valid]
    
    # Component 1: Best-fit (negative remaining, so least remaining = highest score)
    # Normalize to [0, 1] range for combination
    max_rem = np.max(valid_remaining)
    if max_rem > 1e-10:
        norm_rem = valid_remaining / max_rem  # 0 = full, 1 = empty
    else:
        norm_rem = np.zeros_like(valid_remaining)
    
    # Best-fit score: prefer low remaining (high fill)
    best_fit_score = 1.0 - norm_rem  # 1.0 = bin is full, 0.0 = bin is empty
    
    # Component 2: Fill-level modulation
    # We want to prefer bins with moderate fill (0.3-0.7) to consolidate
    # but not overpack. Use a Gaussian-like bump centered at 0.5.
    fill = 1.0 - norm_rem  # Fill level: 0 = empty, 1 = full
    # Gaussian bump: exp(-((fill - 0.5)^2) / (2 * sigma^2))
    sigma = 0.25
    bump_score = np.exp(-((fill - 0.5) ** 2) / (2 * sigma ** 2))
    
    # Component 3: Penalty for very full bins (to leave room for future large items)
    # This is already somewhat captured by best-fit, but add explicit penalty
    # for bins that would exceed 90% fill
    penalty = np.where(fill > 0.9, (fill - 0.9) * 10.0, 0.0)
    
    # Combine: weighted sum
    # best_fit: strong signal to pack tightly
    # bump: moderate preference for consolidation
    # penalty: avoid overpacking
    w_best_fit = 1.0
    w_bump = 0.3
    w_penalty = 1.5
    
    combined = w_best_fit * best_fit_score + w_bump * bump_score - w_penalty * penalty
    
    scores[valid] = combined
    return scores
# EVOLVE-BLOCK-END
