import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    rem = bins - item
    # Threshold below which remaining space is considered "wasteful"
    threshold = 10.0  # tune this
    # Penalty for small remainders
    penalty = np.where(rem < threshold, (threshold - rem) / threshold, 0.0)
    # Score: prefer smaller rem, but penalize very small rem
    scores = -(rem + penalty * threshold)
    return scores
