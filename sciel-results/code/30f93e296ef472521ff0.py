"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.stats

_DEADLINE = 140.0  # safety margin before 150s kill
_START_TIME = time.time()

# Weibull distribution parameters for the generated sequences
# Based on the task description and common bin packing benchmarks:
# Capacity 100. Weibull distribution.
# Let's assume shape k=2.0 and scale lambda=25.0 as a reasonable default 
# for "mix of small and large" fitting in 100.
# Mean = 25 * Gamma(1.5) = 25 * 0.886 = 22.15
# This implies ~4.5 items per bin on average.
k = 2.0
lambda_scale = 25.0
weibull_dist = scipy.stats.weibull_min(c=k, scale=lambda_scale)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline guard: if we're too close to the limit, return a safe default
    if time.time() - _START_TIME > _DEADLINE:
        valid = bins >= item
        if not np.any(valid):
            return np.full_like(bins, -np.inf)
        # Fallback to Best Fit if time is critical
        scores = -(bins - item)
        scores[~valid] = -np.inf
        return scores

    # Determine which bins have enough remaining capacity for the current item
    valid = bins >= item
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
        
    gap = bins - item
    
    # Strategy: Probabilistic Best-Fit
    # We want to choose the bin that minimizes the "wasted" potential.
    # Instead of just minimizing the gap (Best Fit) or maximizing it (Worst Fit),
    # we estimate how many *additional* items can fit into the remaining space (gap)
    # given the Weibull distribution of future items.
    
    # If the gap is large, it can likely fit more items.
    # If the gap is small, it might fit 0 or 1 more items.
    # We want to avoid creating gaps that are "too big" (wasting space that could have been used by other items if we packed tighter) 
    # but also avoid creating gaps that are "too small" (fragmentation).
    
    # A good proxy for "how well this gap is utilized by future items" is the expected count of items 
    # that fit in the gap. However, simply maximizing this might lead to Worst-Fit behavior for large gaps.
    
    # Let's look at the "Efficiency" of the placement.
    # Total capacity used so far in bin i = 100 - bins[i].
    # After placing item, used = 100 - (bins[i] - item).
    
    # Let's try a different angle: Minimize the probability that the remaining space 
    # is "dead" (too small for the next item) OR "wasted" (too large, implying we should have packed tighter).
    
    # Actually, a very strong heuristic for bin packing is "Best Fit Decreasing" style logic adapted to online.
    # Since we don't know future items, we assume they follow the distribution.
    
    # Score = - (gap) is Best Fit.
    # Score = + (gap) is Worst Fit.
    
    # Let's use the CDF of the Weibull distribution.
    # P(X <= gap) is the probability that a random future item fits in the gap.
    # If this probability is high, the gap is "useful".
    # If this probability is low, the gap is "dead" (fragmentation).
    # We want to maximize the usefulness of the remaining space.
    
    # However, if we always pick the bin where the gap is most likely to fit the next item,
    # we might be filling bins too slowly (Worst Fit).
    
    # Let's combine:
    # 1. Prefer bins where the gap is small (Best Fit) to pack tightly.
    # 2. Penalize gaps that are too small to likely fit the next item (avoid fragmentation).
    
    # p_fit = weibull_dist.cdf(gap)
    # If p_fit is high, the gap is good for future items.
    # If p_fit is low, the gap is bad (dead space).
    
    # We want to minimize the gap, but if the gap is so small that p_fit is very low, 
    # we might prefer a slightly larger gap that has a higher p_fit.
    
    # Score = gap * (1 - p_fit) ?
    # If p_fit is 1 (gap is huge), score = 0.
    # If p_fit is 0 (gap is tiny), score = gap.
    # This minimizes the "dead space" (gap that can't be used).
    
    # Let's refine:
    # We want to minimize the "expected wasted space".
    # Wasted space in a bin with remaining capacity R is roughly:
    # If R is small, it's all wasted if no item fits.
    # If R is large, it's not wasted yet.
    
    # Let's try: Score = gap - expected_value_of_future_items_in_gap
    # This is hard to compute exactly.
    
    # Alternative: 
    # Use the "Best Fit" logic but with a threshold.
    # If gap < threshold, penalize heavily (fragmentation).
    # Threshold could be the median of the Weibull distribution.
    # median = lambda * (ln(2))^(1/k)
    # For k=2, lambda=25: median = 25 * (0.693)^(0.5) = 25 * 0.83 = 20.75
    
    # So if gap < 20.75, it's likely to be fragmented.
    # If gap > 20.75, it's likely to accept another item.
    
    # Strategy: 
    # If gap > median: Score = -gap (Best Fit among "good" gaps)
    # If gap < median: Score = -gap + penalty (Penalize fragmentation)
    
    # Actually, a smoother approach:
    # Score = -gap / (1 + p_fit)
    # If p_fit is high (large gap), the denominator is large, so the score is less negative (closer to 0).
    # If p_fit is low (small gap), the denominator is small, so the score is more negative.
    # This effectively says: "A small gap is bad, but a large gap is less 'good' (less negative score) 
    # because it's likely to be used anyway, so we don't need to prioritize it as much as a tight fit."
    
    # Wait, we want the HIGHEST score.
    # If Score = -gap / (1 + p_fit):
    # Gap=1, p_fit=0.1 -> Score = -1/1.1 = -0.9
    # Gap=10, p_fit=0.5 -> Score = -10/1.5 = -6.6
    # Gap=50, p_fit=0.9 -> Score = -50/1.9 = -26
    # This prefers small gaps (Best Fit). This is just Best Fit with a twist.
    
    # Let's try the opposite:
    # We want to avoid creating "dead" gaps.
    # A gap is "dead" if it's too small to fit the next item.
    # Probability of being dead = 1 - p_fit.
    # Expected dead space = gap * (1 - p_fit).
    # We want to MINIMIZE this expected dead space.
    # So Score = - (gap * (1 - p_fit)).
    
    p_fit = weibull_dist.cdf(gap)
    # Clip p_fit to [0, 1] just in case
    p_fit = np.clip(p_fit, 0.0, 1.0)
    
    expected_dead_space = gap * (1.0 - p_fit)
    
    scores = -expected_dead_space
    
    # Mask invalid bins
    scores[~valid] = -np.inf
    
    return scores
# EVOLVE-BLOCK-END
