import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Uses a hybrid heuristic combining Best Fit with a Gaussian bump
    centered at a small fraction of the item size to encourage
    'slightly loose' fits that may reduce fragmentation over time.
    """
    remaining = bins - item
    
    # Initialize scores with -inf for invalid bins
    scores = np.full_like(remaining, -np.inf, dtype=np.float64)
    
    # Only consider bins with enough remaining capacity
    valid = remaining >= 0
    
    if not np.any(valid):
        return scores
    
    r = remaining[valid]
    
    # Primary: Best Fit (smaller remaining is better)
    # We want to maximize score, so we negate the remaining capacity.
    # However, the scale of 'r' can vary. Let's normalize or just use it directly.
    # The current program uses -r.
    base_score = -r
    
    # Secondary: Gaussian bump
    # The current program centers at 0.1 * item with sigma 0.05 * item.
    # This encourages a small gap, avoiding both exact fits (which might be fragile if estimates are off)
    # and large gaps (waste).
    center = 0.1 * item
    sigma = 0.05 * item + 1e-10
    bump = np.exp(-((r - center) ** 2) / (2.0 * sigma ** 2))
    
    # Tertiary: Large Gap Penalty (from Inspiration 2)
    # Penalize bins that are much larger than the item.
    # This helps avoid filling a huge bin with a tiny item, which wastes space.
    # The base_score (-r) already penalizes large gaps, but linearly.
    # A quadratic or super-linear penalty might be stronger for very large gaps.
    # Let's add a penalty for gaps > 2 * item.
    gap_ratio = r / (item + 1e-10)
    # Penalty starts when gap_ratio > 2.0
    # The penalty should be significant enough to override the bump if the gap is huge.
    # Since base_score is -r, a gap of 2*item gives -2*item.
    # A gap of 10*item gives -10*item.
    # The bump is at most 1.0.
    # So base_score dominates for large gaps.
    # However, if item is small, -r might not be "large" enough in absolute terms
    # compared to the bump? No, bump is bounded [0,1].
    # If item is 0.1, r=2.0 (20x item), base=-2.0. bump ~ 0.
    # If item is 0.1, r=0.1 (1x item), base=-0.1. bump ~ exp(-((0.1-0.01)^2)/(2*0.005^2)) ...
    # center=0.01, sigma=0.005.
    # r=0.1: (0.1-0.01)^2 = 0.0081. 2*sigma^2 = 5e-5.
    # exp(-162) ~ 0.
    # So bump is very narrow.
    
    # Let's refine the parameters.
    # The current program uses weight 0.5 for the bump.
    # base_score range: [-max_r, 0].
    # If max_r is large, base_score dominates.
    # If max_r is small (tight packing), bump might have more influence relative to base differences.
    
    # Let's try to combine them more carefully.
    # Score = -r + weight * bump - penalty
    
    # The penalty from inspiration 2 was:
    # large_gap_penalty = np.maximum(gap_ratio - 2.0, 0.0) * 0.5
    # This is linear in gap_ratio.
    
    # Let's stick to the core of the current program but ensure the bump is effective.
    # The current program's bump is very narrow (sigma=0.05*item).
    # This means it only rewards a very specific range of remaining capacity.
    # If the remaining capacity doesn't fall in that narrow band, the bump is ~0.
    # Then it's just Best Fit.
    
    # Maybe widen the sigma slightly to make the "preference" more robust?
    # Or adjust the center?
    
    # Let's look at the "Slightly Loose" idea.
    # Center = 0.1 * item.
    # Sigma = 0.05 * item.
    # This is a 10% gap with 5% width.
    
    # Let's try a slightly wider sigma to capture more "good" fits.
    sigma = 0.1 * item + 1e-10
    center = 0.1 * item
    
    bump = np.exp(-((r - center) ** 2) / (2.0 * sigma ** 2))
    
    # Weight: How much does the bump matter?
    # If weight is too high, we might pick a bin with r=0.1*item over r=0.01*item
    # even though r=0.01*item is a better fit.
    # base_score difference: -0.1*item - (-0.01*item) = -0.09*item.
    # bump difference: 1.0 - 0 = 1.0.
    # If weight > 0.09*item, we prefer the looser fit.
    # If item is small (e.g., 0.1), 0.09*item = 0.009.
    # So weight > 0.009 is enough to flip the decision.
    # The current weight is 0.5. This is very high.
    # It effectively means: "If a bin has ~10% remaining capacity, pick it, even if another bin has 1% remaining."
    # This is a strong preference for "slightly loose" fits.
    
    # Is this good?
    # It might reduce fragmentation by leaving small gaps that can be filled by future small items.
    # But it wastes capacity on larger items.
    
    # Let's try a moderate weight, e.g., 0.1 or 0.2.
    weight = 0.1
    
    combined = base_score + weight * bump
    
    scores[valid] = combined
    return scores
