"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid = remaining >= 0
    
    # Best Fit with penalty for very small remainders
    # Small remainders (< 5) create unusable gaps that waste capacity
    # Use a smooth penalty: as remaining approaches 0-5, add a penalty
    # that grows quadratically to discourage these tight fits
    
    # Base score: prefer smaller remainders (Best Fit)
    base_score = -remaining
    
    # Penalty for very small remainders: 
    # When remaining < 5, add a penalty that increases as remaining -> 0
    # This uses a smooth function: penalty = max(0, (5 - remaining))^2 / 5
    # At remaining=5: penalty=0, at remaining=0: penalty=5
    # This makes tight fits (remaining < 5) less attractive
    
    small_gap_penalty = np.maximum(0, 5.0 - remaining) ** 2 / 5.0
    
    # Combined score: prefer small remainders but penalize unusable gaps
    score = np.where(valid, base_score - small_gap_penalty, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
