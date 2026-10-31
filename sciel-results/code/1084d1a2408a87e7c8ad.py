"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_CAPACITY = 100.0


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Hybrid scoring: smooth transition from Best-Fit (small items) to
    Worst-Fit (large items) based on item size relative to bin capacity.
    
    - Small items: consolidate into nearly-full bins (Best-Fit)
    - Large items: spread across bins to keep them open (Worst-Fit)
    - Smooth sigmoid transition around t = item/capacity ≈ 0.5
    """
    remaining_after = bins - item
    valid_mask = remaining_after >= 0

    scores = np.full_like(bins, -np.inf, dtype=np.float64)

    if not np.any(valid_mask):
        return scores

    valid_remaining = remaining_after[valid_mask]

    # Relative item size
    t = item / _CAPACITY

    # Sigmoid weight: w ≈ 1 for small items (Best-Fit), w ≈ 0 for large (Worst-Fit)
    # Sharpness k controls transition width; t0 is the midpoint
    k = 20.0
    t0 = 0.5
    w = 1.0 / (1.0 + np.exp(k * (t - t0)))

    # score = w * (-remaining_after) + (1-w) * (remaining_after)
    #       = (2w - 1) * remaining_after  ... wait, let me recheck
    # Actually: w*(-r) + (1-w)*(r) = r*(1 - w - w) = r*(1 - 2w)
    # When w=1: r*(1-2) = -r  → Best-Fit ✓
    # When w=0: r*(1-0) = +r  → Worst-Fit ✓
    coeff = 1.0 - 2.0 * w
    scores_valid = coeff * valid_remaining

    scores[valid_mask] = scores_valid
    return scores
# EVOLVE-BLOCK-END
