import numpy as np
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining_after = bins - item
    feasible = remaining_after >= 0

    if not np.any(feasible):
        return np.full(len(bins), -np.inf)

    scores = np.full(len(bins), -np.inf)
    
    # Use a non-linear score that strongly favors tight fits
    # score = -remaining_after + bonus * (1 + 1/remaining_after)
    # This gives a higher score to bins with smaller remaining_after
    # The 1/remaining_after term creates a strong non-linear bonus
    
    ra = remaining_after[feasible]
    # Avoid division by zero
    ra_safe = np.maximum(ra, 0.1)
    scores[feasible] = -ra + 5.0 * (1.0 / ra_safe)
    
    return scores
