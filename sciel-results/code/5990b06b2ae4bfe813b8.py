import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    rem = bins - item
    C = 100.0
    # Base bimodal score (measured 1.02)
    # Add slight preference for larger residuals (Worst Fit bias)
    # to keep bins open for potential larger items
    score = -rem * (C - rem) + 0.2 * rem
    # Ensure invalid bins (if any) are scored low
    score = np.where(rem >= 0, score, -1e9)
    return score
