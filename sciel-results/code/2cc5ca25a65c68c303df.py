"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for adaptive heuristic
# We track the distribution of remaining capacities to estimate "waste"
# A histogram of bin remainders allows us to calculate the probability that
# a new remainder will be too small to fit future items (fragmentation)
# or too large (inefficiency).

_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # Safety margin

# Adaptive state
# We maintain a histogram of current bin remainders to estimate future item fit.
# Since we don't know the future, we use the current distribution of remainders
# as a proxy for the "scale" of items likely to arrive or fit.
# If a remainder is smaller than the median of current remainders, it's likely "dead".
# If it's much larger, it might be inefficient if we could have packed tighter.

# Simple adaptive parameters
# We use a moving average of the "ideal" remainder target.
# Actually, a simpler structural change: 
# Instead of pure Best-Fit (min remaining), we use a score that penalizes
# remainders that are "too small" to be useful (fragmentation) and "too large" 
# compared to the typical bin usage.
#
# Strategy: "Adaptive Best-Fit with Fragmentation Penalty"
# 1. Estimate the "typical" item size or remaining capacity from the bins themselves.
# 2. Score = -(remaining) - penalty(remaining)
#    where penalty increases if remaining is very small (dead space) 
#    or if remaining is very large (wasted potential, though Best Fit already handles this).
#
# Let's try a different structural approach: 
# **Harmonic Mean Fit**
# The prompt inspiration suggested: "negative of the harmonic mean of their remaining capacity and the incoming item size".
# Harmonic mean H(a,b) = 2ab/(a+b).
# If we maximize H(remaining, item), we favor bins where remaining is proportional to item.
# If remaining >> item, H is small (penalized).
# If remaining << item, H is small (penalized, but feasibility check handles < item).
# If remaining ~ item, H is max.
# This naturally avoids both tiny gaps (fragmentation) and huge gaps (if we were using Worst Fit, but here it's a balance).
#
# Let's implement the Harmonic Mean strategy as suggested in the inspirations, 
# but ensure it's robust.
# Score = - 1 / H(remaining, item) ? No, we want highest score.
# Maximize H(remaining, item).
# H = 2 * remaining * item / (remaining + item)
# This function is maximized when remaining == item.
# It decreases as remaining diverges from item.
# This is a "Proportional Fit" heuristic.

# We will use this directly. It is stateless but structurally different from linear Best-Fit.
# It penalizes large remainders more heavily than Best Fit does (Best Fit just minimizes remaining, 
# so a bin with 1 left is better than 10 left. Harmonic mean: 
# If item=5, rem=5 -> H=5. rem=1 -> H=2*5*1/6=1.66. rem=10 -> H=2*5*10/15=6.66.
# Wait, H is maximized at rem=item. 
# rem=10 (larger than item) gives H=6.66 which is HIGHER than rem=5 (H=5).
# This means it prefers LARGER remainders if they are proportional? 
# No, H(a,b) is symmetric. H(5,10)=6.66. H(5,5)=5. H(5,1)=1.66.
# So it prefers rem=10 over rem=5? That's Worst-Fit behavior for large gaps?
# Let's check the derivative. 
# f(r) = 2*i*r / (r+i). f'(r) = 2*i*(r+i - r)/(r+i)^2 = 2*i^2/(r+i)^2 > 0.
# So H increases with remaining capacity.
# This would select the bin with the LARGEST remaining capacity that is still feasible?
# That's Worst Fit.
# Worst Fit is often competitive with Best Fit.
#
# However, the inspiration said "penalizes both tiny leftover gaps ... and excessively large remaining spaces".
# Harmonic mean does NOT penalize excessively large spaces; it rewards them.
#
# Let's try a different structural change: **Worst Fit Decreasing (WFD)** logic?
# Or **First Fit Decreasing**? We can't sort items.
#
# Let's look at the "Fragmentation" angle again.
# A common improvement over Best Fit is to avoid creating small gaps.
# Score = - (remaining ** 2) ? (Quadratic Best Fit)
# Or Score = - remaining + lambda * (remaining < threshold) ?
#
# Let's try **Worst Fit** (maximize remaining capacity).
# Worst Fit tends to keep bins "open" for larger items, reducing fragmentation of small gaps.
# In online bin packing, Worst Fit is known to perform well, especially when item sizes are uniform or Weibull distributed.
# Best Fit minimizes remaining. Worst Fit maximizes remaining.
# Let's switch to Worst Fit.

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing using a Worst-Fit heuristic.
    
    The score for each bin is its remaining capacity after placing the item.
    Bins that cannot fit the item are assigned a very low score.
    This promotes keeping bins "open" to accommodate larger future items,
    reducing fragmentation compared to Best-Fit.
    """
    if time.time() > _DEADLINE:
        return np.zeros_like(bins, dtype=np.float64)

    remaining = bins - item
    feasible = remaining >= 0
    
    # Initialize scores to a very low value for infeasible bins
    scores = np.full_like(remaining, -1e18, dtype=np.float64)
    
    # For feasible bins, score is the remaining capacity itself (Worst Fit)
    # Higher score = more remaining space = preferred.
    scores[feasible] = remaining[feasible]
    
    return scores
# EVOLVE-BLOCK-END
