"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Strategy: Golden ratio attractor scoring.
# The golden ratio (phi ≈ 0.618) minimizes worst-case waste in sequential subdivision.
# We model each bin's remaining capacity as a 1D spatial coordinate.
# Score = negative Euclidean distance between the resulting remainder and the golden ratio target.
# This creates a stable attractor at 0.618 * capacity, preventing fragmentation.

CAPACITY = 100.0
GOLDEN_RATIO = 0.6180339887498949
TARGET_REM = GOLDEN_RATIO * CAPACITY  # ~61.8

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores bins based on golden ratio attractor.
    bins: array of remaining capacities.
    Returns: array of scores. Higher is better.
    """
    # Filter bins that can fit the item
    valid_mask = bins >= item
    if not np.any(valid_mask):
        scores = np.full_like(bins, -np.inf, dtype=float)
        return scores

    # Calculate remaining capacity after placing item
    rem_after = bins - item

    # Distance from golden ratio target
    # We want to minimize |rem_after - TARGET_REM|
    # So score = -|rem_after - TARGET_REM|
    # Higher score (closer to 0) means closer to the target.

    dist = np.abs(rem_after - TARGET_REM)
    scores = -dist

    # Mask invalid bins
    scores[~valid_mask] = -np.inf

    return scores
# EVOLVE-BLOCK-END
