"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Best-fit-decreasing style: prefer bins where the item fits with minimal excess.
    # Score is negative of remaining capacity after placing item, so highest score
    # (least negative / closest to zero from below) corresponds to tightest fit.
    remaining = bins - item
    # Only consider bins with enough capacity
    valid = remaining >= 0
    # For valid bins, score = -remaining (tighter fit = higher score)
    # For invalid bins, assign a very low score so they are never chosen
    scores = np.full_like(bins, -np.inf, dtype=float)
    scores[valid] = -remaining[valid]
    return scores
# EVOLVE-BLOCK-END
