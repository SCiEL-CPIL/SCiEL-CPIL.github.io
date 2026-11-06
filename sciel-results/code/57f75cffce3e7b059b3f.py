import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid = remaining >= 0
    scores = np.full(len(bins), -np.inf)
    if np.any(valid):
        rem_valid = remaining[valid]
        # Worst Fit: maximize remaining space to reduce fragmentation
        scores[valid] = rem_valid
    return scores
