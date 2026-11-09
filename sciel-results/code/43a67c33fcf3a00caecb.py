import numpy as np
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    gaps = bins - item
    # Mask invalid bins
    valid = gaps >= 0
    scores = np.full_like(bins, -np.inf, dtype=float)
    valid_gaps = gaps[valid]
    # Score: Best Fit term + Small Worst Fit term
    # BF: -gap
    # WF: log(gap + 1) ? No, WF prefers large R, not large gap.
    # R = gap + item.
    # So WF term is log(R + 1) = log(gap + item + 1).
    scores[valid] = -valid_gaps + 0.1 * np.log(valid_gaps + item + 1)
    return scores
