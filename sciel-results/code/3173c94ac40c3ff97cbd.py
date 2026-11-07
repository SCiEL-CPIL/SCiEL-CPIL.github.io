"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 320.0  # safety margin before 330s kill


def heuristics(prizes, weights):
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Per-constraint efficiency
    eff = prizes / (weights + 1e-9)  # (5, 100)
    
    # Scarcity-based constraint weighting:
    # Constraints with less total available weight are tighter, so items
    # efficient in those constraints are more valuable.
    # weight_sum[c] = total weight available for constraint c
    weight_sum = weights.sum(axis=1)  # (5,)
    
    # Scarcity: lower total weight per capacity unit = tighter constraint
    # All capacities are 25, so tighter = smaller weight_sum
    # Use inverse of weight_sum as scarcity weight, normalized
    scarcity = 1.0 / (weight_sum + 1e-9)  # (5,)
    scarcity = scarcity / scarcity.sum()  # normalize to sum to 1
    
    # Weighted geometric mean: log-sum with scarcity weights
    # This emphasizes efficiency in tighter (scarcer) constraints
    log_eff = np.log(eff + 1e-9)  # (5, 100)
    weighted_log = (scarcity[:, None] * log_eff).sum(axis=0)  # (100,)
    weighted_geo = np.exp(weighted_log)  # (100,)
    
    # Scale by total prize to maintain comparable magnitude
    total_prize = prizes.sum(axis=0)  # (100,)
    
    heuristic = weighted_geo * total_prize
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Deadline guard
    if time.time() > _DEADLINE:
        return heuristic
    
    return heuristic
# EVOLVE-BLOCK-END
