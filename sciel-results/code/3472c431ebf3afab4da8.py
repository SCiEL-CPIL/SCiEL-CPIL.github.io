"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Worst-Fit Decreasing heuristic: prefer bins with the most remaining capacity.
    This preserves high-capacity bins for future large items, reducing fragmentation.
    
    The score is the remaining capacity (bins - item) for feasible bins,
    and a very negative value for infeasible bins.
    """
    remaining = bins - item
    # Feasible bins have remaining >= 0; infeasible get a large penalty
    scores = np.where(remaining >= 0, remaining, -1e18)
    return scores
# EVOLVE-BLOCK-END
