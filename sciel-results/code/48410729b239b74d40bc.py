"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining capacity for each bin
    rem = bins - item
    
    # Identify bins that can fit the item
    valid = rem >= 0
    
    # Threshold for considering a remainder "usable"
    # If the remainder is below this, we penalize it heavily
    # to avoid creating small, unusable gaps (fragmentation)
    tau = 5.0
    mult = 10.0
    
    # Calculate penalty for small remainders
    # If rem < tau, penalty is (tau - rem) * mult
    # If rem >= tau, penalty is 0
    diff = tau - rem
    pen = np.where(diff > 0, diff * mult, 0.0)
    
    # Base score is Best Fit: -rem (smaller remainder is better)
    # Subtract penalty to discourage small remainders
    scores = -rem - pen
    
    # Invalidate bins that cannot fit the item
    scores = np.where(valid, scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
