"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Strategy: Fragmentation-aware asymmetric bimodal scoring.
#
# Base: Bimodal score -r * (C - r) which prefers residuals near 0 (tight fit)
# or near C (empty bin). This was the winning family (score 1.0204).
#
# Refinement 1: Asymmetry.
# The pure bimodal score treats r=10 and r=90 equally (both score -900 for C=100).
# However, a tight fit (small r) is generally more valuable than an empty bin
# for small items, because small items are abundant in Weibull distributions.
# Using an empty bin for a small item wastes the "perfect fit" opportunity
# of a tighter bin. So we add a linear penalty -k*r to break symmetry,
# favoring smaller residuals (Best Fit) while retaining the bimodal structure
# that avoids medium residuals.
#
# Refinement 2: Fragmentation-aware penalty.
# The directive suggests penalizing bins where the item is a moderate fraction
# (20-50%) of the remaining capacity. This creates "unusable fragments".
# Let ratio = item / bins.
# If ratio is in [0.2, 0.5], the resulting residual r = bins - item is in [0.5*bins, 0.8*bins].
# This is a "medium" residual that might be hard to fill.
# We can model this as an additional penalty when the residual r is in a "bad" range
# relative to the item size or capacity.
#
# Let's combine:
# Score = -r * (C - r) - k1 * r - k2 * fragmentation_penalty(r, item)
#
# For the fragmentation penalty, we want to penalize cases where the residual
# is not very small and not very large. The bimodal term already does this.
# But the specific "moderate fraction" hint suggests we might want to penalize
# based on the ratio item/bins.
#
# Let's try a smooth penalty for the ratio.
# Let x = item / bins.
# We want to penalize x in [0.2, 0.5].
# A simple smooth function that is 0 outside [0.2, 0.5] and positive inside:
# p(x) = max(0, 0.2 - x) * max(0, x - 0.5) ? No, that's 0 inside.
# We want a bump in [0.2, 0.5].
# p(x) = - (x - 0.35)^2 + (0.15)^2 for x in [0.2, 0.5], else 0.
# This is a parabola peaking at 0.35.
#
# However, calculating this per-bin with numpy masking might be slow or complex.
# Let's stick to a simpler refinement of the bimodal score first, as the
# "fragmentation" effect is already partially captured by the bimodal shape
# (avoiding medium residuals).
#
# The current program uses:
# scores[valid] = -r * (CAPACITY - r) - 0.1 * r
#
# Let's tune the asymmetry coefficient.
# If we increase the linear penalty, it becomes more Best-Fit.
# If we decrease it, it becomes more Worst-Fit (for large r).
#
# Let's try a slightly different structural change:
# Instead of a linear penalty, use a power penalty.
# Score = -r * (C - r) - k * r^alpha
#
# Or, let's look at the "low-utilization bonus" from the cell findings:
# "Stronger bimodal scoring with explicit low-utilization bonus preserves bins
# with large remaining space for future items"
#
# "Low utilization" means the bin is mostly empty (large r).
# So we want to BONUS large r?
# The bimodal term -r(C-r) already gives a high score (near 0) for large r.
# But the linear penalty -0.1*r lowers the score for large r.
# Maybe we should REMOVE the linear penalty or make it smaller?
#
# Let's compare:
# Pure bimodal: score = -r(C-r)
# Asymmetric: score = -r(C-r) - 0.1*r
#
# At r=90 (C=100):
# Pure: -90*10 = -900
# Asym: -900 - 9 = -909
#
# At r=10:
# Pure: -10*90 = -900
# Asym: -900 - 1 = -901
#
# Pure bimodal: r=90 and r=10 are tied.
# Asymmetric: r=10 is better (-901 > -909).
#
# The cell finding says "preserves bins with large remaining space".
# This suggests that using large r (empty bins) is GOOD.
# So we should prefer large r?
# That would mean WORSE FIT?
# But Worst Fit is usually bad.
#
# "Preserves bins with large remaining space for future items"
# This means we should NOT use the large r bins for small items?
# If we don't use them, they stay large r.
# So we should AVOID using large r bins for small items.
# This means we should PENALIZE large r bins when the item is small.
#
# How to implement "penalize large r when item is small"?
# The score should be lower for large r if item is small.
# Current asymmetric score: -r(C-r) - 0.1*r.
# This penalizes large r linearly.
#
# Maybe we should penalize large r MORE if the item is small?
# Score = -r(C-r) - k * (item/C) * r ?
# If item is small, k*(item/C) is small, so penalty is small? No, we want LARGER penalty.
# Score = -r(C-r) - k * (1 - item/C) * r ?
# If item is small, (1-item/C) is large, so penalty is large.
# This penalizes large r more for small items.
#
# Let's try this:
# penalty_factor = 1 - item / CAPACITY
# scores[valid] = -r * (CAPACITY - r) - 0.1 * penalty_factor * r
#
# This is a dynamic penalty based on item size.
#
# Let's also consider the "fragmentation" penalty for moderate ratios.
# We can add a smooth penalty term that is active when item/bins is in [0.2, 0.5].
# But this requires per-bin calculation of item/bins, which is just item / bins.
#
# Let's implement the dynamic penalty first.

CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    start_time = time.time()
    
    # Compute residuals
    residuals = bins - item
    
    # Valid bins are those with non-negative residuals
    valid = residuals >= 0
    
    # Initialize scores to -inf
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    if not np.any(valid):
        return scores

    # Calculate scores for valid bins
    r = residuals[valid]
    b = bins[valid]
    
    # Base bimodal score
    base_score = -r * (CAPACITY - r)
    
    # Dynamic asymmetry penalty:
    # Penalize large residuals more if the item is small.
    # This preserves empty bins for larger items.
    # factor = 1 - item/CAPACITY
    # If item is small (e.g., 1), factor ~ 0.99. Penalty is strong.
    # If item is large (e.g., 90), factor ~ 0.1. Penalty is weak.
    penalty_factor = 1.0 - item / CAPACITY
    # Ensure penalty_factor is not negative (item <= CAPACITY for valid bins)
    penalty_factor = np.maximum(penalty_factor, 0.0)
    
    # Linear penalty term
    linear_penalty = 0.1 * penalty_factor * r
    
    # Fragmentation penalty:
    # Penalize bins where the item is a moderate fraction of the bin.
    # ratio = item / b
    # We want to penalize ratio in [0.2, 0.5].
    # Use a smooth bump function.
    # Let's use a simple quadratic bump centered at 0.35 with width 0.15.
    ratio = item / b
    # Bump: max(0, 0.15 - |ratio - 0.35|)
    # This is 0.15 at ratio=0.35, 0 at ratio=0.2 and 0.5.
    # Square it to make it smoother?
    bump = np.maximum(0.0, 0.15 - np.abs(ratio - 0.35))
    frag_penalty = 50.0 * bump**2  # Scale the penalty
    
    # Total score
    scores[valid] = base_score - linear_penalty - frag_penalty
    
    # Add a tiny tie-breaker for lower indices
    indices = np.arange(len(bins))
    scores[valid] -= indices[valid] * 1e-12
    
    # Deadline check
    if time.time() - start_time > 115:
        # Fallback to simple best fit
        scores = np.where(valid, -residuals, -np.inf)
        
    return scores

# EVOLVE-BLOCK-END
