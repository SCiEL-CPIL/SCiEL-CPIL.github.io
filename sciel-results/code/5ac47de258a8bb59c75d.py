"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 300.0  # seconds wall-clock limit


def heuristics(prizes, weights):
    t0 = time.time()
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    # Ensure positive values to avoid division issues
    prizes = np.maximum(prizes, 1e-9)
    weights = np.maximum(weights, 1e-9)

    capacity = 25.0

    # Per-constraint efficiency: prize/weight for each constraint
    # Shape: (5, 100)
    ratios = prizes / weights

    # Log of ratios for smooth operations
    log_ratios = np.log(ratios)

    # Soft-min across constraints (smooth surrogate for min-ratio)
    neg_log_ratios = -log_ratios
    max_neg = neg_log_ratios.max(axis=0)
    log_sum_exp_neg = max_neg + np.log(np.sum(np.exp(neg_log_ratios - max_neg), axis=0))
    softmin_log = -log_sum_exp_neg

    # Geometric mean of ratios
    log_gm = log_ratios.mean(axis=0)

    # Total prize per item, normalized
    total_prize = prizes.sum(axis=0)
    max_prize = total_prize.max()
    norm_prize = total_prize / (max_prize + 1e-9)

    # Softmin and GM on ratio scale
    softmin_ratio = np.exp(softmin_log)
    gm_ratio = np.exp(log_gm)

    # Bottleneck slack: minimum remaining capacity ratio across all constraints
    # (25 - w_ic) / w_ic for each constraint, then take min across constraints
    slack = (capacity - weights) / weights  # Shape: (5, 100)
    bottleneck_slack = slack.min(axis=0)  # Shape: (100,)
    min_slack = bottleneck_slack.min()
    max_slack = bottleneck_slack.max()
    if max_slack > min_slack:
        norm_slack = (bottleneck_slack - min_slack) / (max_slack - min_slack)
    else:
        norm_slack = np.ones_like(bottleneck_slack)

    # Per-constraint "how much of capacity this item uses"
    # An item that uses very little of a constraint is "safe" on that constraint
    # Compute the maximum fraction of any single constraint this item consumes
    # (i.e., the tightest constraint for this item)
    frac_used = weights / capacity  # (5, 100), each in (0, 1)
    max_frac = frac_used.max(axis=0)  # (100,) - the most constraining dimension
    # Invert: items that use less of their tightest constraint are better
    # Normalize to [0, 1]
    min_frac = max_frac.min()
    max_frac_max = max_frac.max()
    if max_frac_max > min_frac:
        norm_frac_inv = 1.0 - (max_frac - min_frac) / (max_frac_max - min_frac)
    else:
        norm_frac_inv = np.ones_like(max_frac)

    # Primary metric: bottleneck slack * total prize
    # This directly prioritizes items that fit most easily into the tightest constraints
    # while also rewarding high total prize
    feasibility_margin = norm_slack * norm_frac_inv  # Combined feasibility score in [0,1]
    
    # Heuristic: emphasize feasibility margin strongly, combined with ratio efficiency
    heuristic = (
        (0.5 + 0.5 * norm_prize)
        * (0.3 + 0.7 * feasibility_margin)
        * (softmin_ratio * gm_ratio)
    )

    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # Deadline guard
    if time.time() - t0 > _DEADLINE - 5.0:
        return prizes.sum(axis=0) / (weights.sum(axis=0) + 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
