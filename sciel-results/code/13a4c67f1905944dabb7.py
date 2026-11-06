"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Quadratic best-fit: prefer bins that leave the least remaining capacity,
    # with a stronger penalty for large gaps than linear best-fit.
    remaining = bins - item
    return -(remaining ** 2)
# EVOLVE-BLOCK-END
