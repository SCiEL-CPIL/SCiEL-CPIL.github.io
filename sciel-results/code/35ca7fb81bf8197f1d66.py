"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Worst-Fit: prefer bins that leave the most remaining capacity.
    # This helps keep large gaps available for potentially larger future items,
    # which is often beneficial for skewed distributions like Weibull.
    scores = bins - item
    return scores
# EVOLVE-BLOCK-END
