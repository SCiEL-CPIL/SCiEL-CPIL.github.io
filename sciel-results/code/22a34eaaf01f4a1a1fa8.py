"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 300.0


def heuristics(prizes, weights):
    t0 = time.time()
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    # Ensure shapes are (5, 100)
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        return np.ones(100)

    # Clip to avoid zero/negative issues
    prizes = np.maximum(prizes, 1e-9)
    weights = np.maximum(weights, 1e-9)

    # Per-constraint ratio: prize / weight for each constraint
    ratios = prizes / weights  # shape (5, 100)

    # Log of ratios for smooth operations
    log_ratios = np.log(ratios)

    # Soft-min across constraints using log-sum-exp trick
    # softmin(x, T) = -T * log(sum(exp(-x/T)))
    # Lower T = sharper (closer to true min)
    T = 0.05
    neg_scaled = -log_ratios / T
    # For numerical stability, subtract max
    max_neg = np.max(neg_scaled, axis=0, keepdims=True)
    softmin_log = -T * (max_neg[0] + np.log(np.sum(np.exp(neg_scaled - max_neg), axis=0)))

    # This gives a smooth approximation of the log of the minimum ratio
    # Higher softmin_log = better worst-case efficiency

    # Also compute the mean log ratio (average efficiency)
    mean_log_ratio = np.mean(log_ratios, axis=0)

    # Combine: weight the soft-min (bottleneck) heavily, add some mean efficiency
    # The bottleneck matters most for feasibility
    combined = 2.0 * softmin_log + 0.5 * mean_log_ratio

    # Sharpening: raise to a power to amplify differences
    # First, shift to make values positive for power operation
    min_val = np.min(combined)
    shifted = combined - min_val + 1.0  # all >= 1.0

    # Power sharpening (higher power = more differentiation)
    sharpened = shifted ** 3.0

    # Multiply by total prize to reward high-value items
    total_prize = np.sum(prizes, axis=0)
    heuristic = sharpened * total_prize

    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
