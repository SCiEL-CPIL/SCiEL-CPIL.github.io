"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Adaptive Best-Fit with consolidation bias.
    For large items, strongly prefer bins where the item fits tightly (Best-Fit).
    For small items, use a smoother preference that still favors tighter fits
    but is less aggressive, to avoid fragmentation from over-consolidation.
    
    We use a sigmoid-based weighting on the "tightness" of the fit.
    Tightness = 1 / (remaining_after + eps). Higher tightness = better fit.
    The sigmoid maps tightness to a [0, 1] weight that blends between
    a neutral score and a strong best-fit score based on item size relative to capacity.
    """
    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores

    remaining_after = bins - item
    feasible = remaining_after >= 0

    if not np.any(feasible):
        return np.full(len(bins), -np.inf)

    scores = np.full(len(bins), -np.inf)

    # For feasible bins, compute a base best-fit score
    # Primary: minimize remaining capacity after placement
    base_score = -remaining_after[feasible]

    # Compute tightness: how well the item fits (higher = tighter)
    eps = 1e-9
    tightness = 1.0 / (remaining_after[feasible] + eps)

    # Normalize tightness to [0, 1] range for sigmoid input
    # The max tightness would be 1/item (when remaining_after ~ 0)
    # The min tightness for feasible bins is 1/(bins_max - item)
    # We'll use a relative scale
    max_tightness = 1.0 / item if item > 0 else 1.0
    normalized_tightness = tightness / (max_tightness + eps)
    normalized_tightness = np.clip(normalized_tightness, 0, 1)

    # Determine blend factor based on item size relative to capacity (100)
    # For large items (> 25), strongly prefer best-fit
    # For small items (< 10), use a gentler preference
    # Use a smooth sigmoid transition
    item_ratio = item / 100.0  # relative to capacity
    # Sigmoid centered at 0.25 (25% of capacity), with steepness 8
    blend = 1.0 / (1.0 + np.exp(-8.0 * (item_ratio - 0.25)))
    
    # blend ~ 0 for small items, ~ 1 for large items
    # Score = base_score * blend + (gentler_score) * (1 - blend)
    # Gentler score: still prefer tighter fits but with less weight
    # Use a softer penalty: -sqrt(remaining_after) instead of -remaining_after
    gentler_score = -np.sqrt(remaining_after[feasible] + eps)
    
    # Combine
    combined = blend * base_score + (1 - blend) * gentler_score
    
    # Add a small consolidation bonus: prefer bins that are already partially filled
    # This is captured by the tightness term
    consolidation_bonus = 0.005 * normalized_tightness
    
    scores[feasible] = combined + consolidation_bonus
    
    return scores
# EVOLVE-BLOCK-END
