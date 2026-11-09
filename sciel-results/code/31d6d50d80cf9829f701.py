"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Score bins by negative fractional remaining capacity (waste ratio).
    # This is a change of variables: instead of penalizing absolute remaining
    # space, penalize the fraction of the bin that will remain empty after
    # placing the item. This makes the heuristic scale-aware and rewards
    # bins where the item fills a larger proportion of available space.
    mask = bins >= item
    scores = np.full_like(bins, -np.inf)
    if np.any(mask):
        # Fraction of remaining capacity that will be wasted
        waste_frac = (bins[mask] - item) / np.maximum(bins[mask], 1e-9)
        scores[mask] = -waste_frac
    return scores
# EVOLVE-BLOCK-END
