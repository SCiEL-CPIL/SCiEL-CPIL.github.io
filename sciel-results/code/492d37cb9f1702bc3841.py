"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_START_TIME = time.time()
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 5.0  # seconds before deadline to stop


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Returns an array of scores; the item goes to the highest-scoring bin.
    
    Strategy: Best-Fit with adaptive fragment penalty using smooth soft-min surrogate.
    - Base score: -remaining (prefer bins with least remaining capacity)
    - Penalty: smooth penalty for creating very small residuals (fragments)
      that are unlikely to be useful for future items
    - Uses log-sum-exp based soft-min to create a differentiable surrogate
      that smoothly blends between best-fit and fragment-avoidance
    """
    # Deadline guard: if we're running out of time, fall back to simple best-fit
    if time.time() > _START_TIME + DEADLINE - SAFETY_MARGIN:
        remaining = bins - item
        valid_mask = remaining >= -1e-12
        scores = np.full(len(bins), -np.inf)
        if np.any(valid_mask):
            scores[valid_mask] = -remaining[valid_mask]
        return scores
    
    remaining = bins - item
    
    # Only bins with enough remaining capacity
    valid_mask = remaining >= -1e-12
    
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid_mask):
        return scores
    
    valid_remaining = remaining[valid_mask]
    n_valid = len(valid_remaining)
    
    if n_valid == 0:
        return scores
    
    # Base best-fit score: -remaining (higher = less remaining = preferred)
    base_score = -valid_remaining
    
    # Adaptive fragment penalty using smooth soft-min:
    # We want to avoid creating residuals that are too small to be useful.
    # Use a smooth approximation to the minimum of (residual, threshold) 
    # to create a differentiable penalty that peaks at small residuals.
    
    # Threshold for "useful" residual: scale with item size
    threshold = max(item * 0.15, 1.0)  # minimum useful residual
    
    # Smooth soft-min of residual and threshold using log-sum-exp:
    # soft_min(a, b) ≈ -log(exp(-a/τ) + exp(-b/τ)) * τ for small τ
    # This gives a smooth transition between min(a,b) and max(a,b)
    tau = 0.5  # smoothing parameter
    
    # Compute soft-min(residual, threshold) for each valid bin
    # When residual < threshold, soft_min ≈ residual (small fragment)
    # When residual > threshold, soft_min ≈ threshold (no penalty)
    residual = valid_remaining
    
    # Soft-min computation: -tau * log(exp(-residual/tau) + exp(-threshold/tau))
    # To avoid overflow, use log-sum-exp trick
    a = -residual / tau
    b = -threshold / tau
    max_ab = np.maximum(a, b)
    soft_min = -tau * (max_ab + np.log(np.exp(a - max_ab) + np.exp(b - max_ab)))
    
    # Penalty: quadratic in (threshold - soft_min) when soft_min < threshold
    # This creates a smooth penalty that is 0 when residual >= threshold
    # and increases as residual decreases toward 0
    diff = threshold - soft_min
    diff_clipped = np.maximum(diff, 0.0)
    
    alpha = 1.5  # strength of fragment penalty
    
    # Smooth penalty: alpha * (diff_clipped / threshold)^2
    penalty = alpha * (diff_clipped / threshold) ** 2
    
    # Combine: base score minus penalty
    score_valid = base_score - penalty
    
    scores[valid_mask] = score_valid
    
    return scores
# EVOLVE-BLOCK-END
