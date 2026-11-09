"""Improved heuristic for ACO bin packing: complementarity + slack + tightness."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Return an n x n matrix of attractiveness for placing item j right after item i.
    
    Strategy: combine three signals:
    1. Complementarity: reward pairs where residual = capacity - (d_i + d_j) is a near-integer
       multiple of the smallest item size (structural fit).
    2. Slack minimization: prefer pairs that leave less wasted space.
    3. Tightness: prefer pairs that use a larger fraction of capacity (but still fit).
    
    Pairs that don't fit get 1e-9.
    """
    demands = np.asarray(demands, dtype=np.float64)
    n = len(demands)
    
    # Precompute
    d = demands  # shape (n,)
    pair_sum = d[:, None] + d[None, :]  # shape (n, n)
    fits = pair_sum <= capacity  # boolean mask
    
    # Residual space after placing i and j together
    residual = capacity - pair_sum  # shape (n, n)
    
    # Smallest item size in this instance
    min_size = float(np.min(d))
    max_size = float(np.max(d))
    mean_size = float(np.mean(d))
    
    # --- Signal 1: Complementarity ---
    # Reward residual being a near-integer multiple of min_size
    # k = residual / min_size, want k to be close to an integer
    # Use a smooth penalty: exp(-alpha * distance_to_nearest_integer^2)
    k = residual / min_size
    nearest_int = np.round(k)
    dist_to_int = np.abs(k - nearest_int)
    # Gaussian kernel around integer multiples
    sigma = 0.3  # width of tolerance
    complementarity = np.exp(-dist_to_int ** 2 / (2 * sigma ** 2))
    
    # --- Signal 2: Slack minimization ---
    # Lower residual is better (less wasted space)
    # Normalize: residual ranges from 0 to capacity
    slack_score = 1.0 / (1.0 + residual / (capacity * 0.1))
    # This gives ~1.0 for zero residual, ~0.5 for 10% capacity residual, etc.
    
    # --- Signal 3: Tightness / utilization ---
    # Prefer pairs that use a larger fraction of the bin
    utilization = pair_sum / capacity
    tightness = utilization  # higher utilization = more attractive
    
    # --- Combine signals ---
    # Weighted combination: complementarity gets highest weight for structural fit,
    # then slack, then tightness
    w_comp = 0.5
    w_slack = 0.3
    w_tight = 0.2
    
    # Normalize each signal to [0, 1] range for fair combination
    comp_min, comp_max = complementarity.min(), complementarity.max()
    if comp_max > comp_min:
        comp_norm = (complementarity - comp_min) / (comp_max - comp_min)
    else:
        comp_norm = np.ones_like(complementarity)
    
    slack_min, slack_max = slack_score.min(), slack_score.max()
    if slack_max > slack_min:
        slack_norm = (slack_score - slack_min) / (slack_max - slack_min)
    else:
        slack_norm = np.ones_like(slack_score)
    
    tight_min, tight_max = tightness.min(), tightness.max()
    if tight_max > tight_min:
        tight_norm = (tightness - tight_min) / (tight_max - tight_min)
    else:
        tight_norm = np.ones_like(tightness)
    
    combined = w_comp * comp_norm + w_slack * slack_norm + w_tight * tight_norm
    
    # Apply fit mask: non-fitting pairs get 1e-9
    result = np.where(fits, combined, 1e-9)
    
    # Ensure all values are positive and finite
    result = np.where(np.isfinite(result) & (result > 0), result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
