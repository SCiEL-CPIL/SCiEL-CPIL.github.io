"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin for placing the given item.
    Higher score = preferred bin.
    Bins with insufficient capacity get -inf.
    
    Strategy: Worst-Fit - prefer bins with the most remaining capacity
    after placing the item, to preserve large bins for future large items.
    """
    remaining = bins - item
    feasible = remaining >= 0
    scores = np.where(feasible, remaining, -np.inf)
    return scores
# EVOLVE-BLOCK-END
