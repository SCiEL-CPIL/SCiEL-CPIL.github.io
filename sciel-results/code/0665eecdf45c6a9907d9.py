"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # Total prize per item (sum across constraints)
    total_prize = prizes.sum(axis=0)
    
    # Smooth max of weights across constraints using log-sum-exp
    # weights shape: (5, 100) - 5 constraints, 100 items
    # We want smooth max across constraints (axis=0) for each item
    max_w = weights.max(axis=0)  # shape: (100,)
    # For numerical stability
    shifted = weights - max_w[np.newaxis, :]  # shape: (5, 100)
    log_sum_exp_vals = max_w + np.log(np.exp(shifted).sum(axis=0) + 1e-15)
    
    # This is a smooth approximation of max(weights) per item
    # Higher temperature (larger beta) makes it closer to hard max
    # We use the log-sum-exp directly as a smooth max
    
    # Heuristic: prize per unit of bottleneck weight
    # Items with high prize and low bottleneck weight are most attractive
    heuristic = total_prize / (log_sum_exp_vals + 1e-9)
    
    # Ensure positive values and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
