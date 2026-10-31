"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Worst Fit: prefer bins that leave the most remaining capacity.
    # This spreads items out, which can be effective for certain distributions.
    return bins - item
# EVOLVE-BLOCK-END
