"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Computes attractiveness scores based on total prize and bottleneck slack.
    
    The heuristic is defined as:
    H_i = (sum of prizes for item i) * (25 / max_weight_i - 1)
    
    Where max_weight_i is the maximum weight of item i across all constraints.
    This prioritizes items that have high total value and are "small" in the
    tightest constraint dimension, effectively maximizing the feasibility margin.
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Total prize per item (shape: 100,)
    total_prize = prizes.sum(axis=0)
    
    # Maximum weight per item across constraints (shape: 100,)
    # This identifies the most constraining dimension for each item
    max_weight = weights.max(axis=0)
    
    # Calculate bottleneck slack
    # slack = 25 / max_weight - 1
    # Using epsilon to prevent division by zero, though weights are in [0, 1]
    slack = 25.0 / (max_weight + 1e-9) - 1.0
    
    # Combine prize and slack
    h = total_prize * slack
    
    # Ensure all values are positive as required by the task contract
    # "Heuristic entries that are not positive become 1e-9"
    h = np.where(h > 0, h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
