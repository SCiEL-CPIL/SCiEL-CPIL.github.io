import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores

    remaining_after = bins - item
    feasible = remaining_after >= 0

    if not np.any(feasible):
        return np.full(len(bins), -np.inf)

    scores = np.full(len(bins), -np.inf)
    
    # Thresholded Best-Fit
    # Penalize bins where the remaining space is very small (< THRESHOLD)
    # This avoids creating "dead" gaps that cannot be used by future items.
    THRESHOLD = 5.0
    PENALTY = 10.0
    
    rem = remaining_after[feasible]
    
    # Base score: Best-Fit (minimize remaining)
    base_score = -rem
    
    # Penalty: if rem < THRESHOLD, subtract PENALTY
    # We use a smooth penalty to avoid discontinuities, or a hard one?
    # Let's try a hard penalty first for simplicity, or a linear ramp.
    # Ramp: penalty = PENALTY * max(0, THRESHOLD - rem) / THRESHOLD
    # If rem=0, penalty=PENALTY. If rem=THRESHOLD, penalty=0.
    penalty = PENALTY * np.maximum(0, THRESHOLD - rem) / THRESHOLD
    
    scores[feasible] = base_score - penalty
    
    return scores
