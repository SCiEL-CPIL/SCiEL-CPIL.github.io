"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Compute remaining capacity after placing the item in each bin.
    remaining = bins - item
    
    # Only consider bins that have enough capacity.
    feasible_mask = remaining >= 0
    if not np.any(feasible_mask):
        # If no bin can fit the item, return a large negative score for all
        # (the evaluator should handle this, but we need to return something).
        return np.full_like(bins, -np.inf, dtype=float)
    
    # We want to maximize the score, so we want to pick the bin that leads to
    # the best long-term packing. A common heuristic is "Best Fit Decreasing"
    # or variations that consider future items.
    #
    # The task mentions Weibull sequences. Weibull distributions with shape parameter k < 1
    # are heavy-tailed (more small items, fewer large ones, but with high variance).
    # Usually, online bin packing with small items benefits from keeping bins "full"
    # to avoid fragmentation, but also from not filling a bin too much if a large item
    # might arrive.
    #
    # Let's try a score based on the "waste" created.
    # Score = - (remaining capacity)^alpha
    # This prefers bins with less remaining capacity (Best Fit).
    #
    # However, we can be smarter. If the remaining capacity is very small, it's essentially useless.
    # If the remaining capacity is large, it might be wasted if no item fits.
    #
    # Let's look at the distribution of items.
    # If items are Weibull, the mean is known.
    #
    # A robust heuristic for online bin packing is "Next Fit" (bad), "First Fit" (okay), "Best Fit" (good).
    # "Worst Fit" is sometimes competitive for certain distributions.
    #
    # Let's try a weighted combination of Best Fit and Worst Fit, or a non-linear Best Fit.
    #
    # Score = -remaining^2  (quadratic best fit)
    # Or Score = -remaining (linear best fit)
    #
    # Let's try to incorporate the idea of "gap utilization".
    # If we place item in bin, the new remaining is r.
    # The probability that a future item fits in r is P(X <= r).
    # For Weibull, CDF is 1 - exp(-(x/scale)^shape).
    # We don't know the exact parameters, but we can assume a typical shape.
    #
    # Actually, a simpler and often very effective strategy is to minimize the "dead space".
    # Dead space = remaining capacity that is too small for the average item.
    #
    # Let's try a score that penalizes remaining capacity that is less than the item size (fragmentation)
    # and also penalizes large remaining capacity (waste).
    #
    # Hypothesis: Best Fit (minimize remaining) is strong. Let's try to refine it.
    # What if we prefer bins where the remaining capacity is just slightly larger than 0?
    #
    # Let's try: score = -remaining, but with a twist.
    # If remaining is very small, it's good.
    #
    # Let's try "Best Fit" with a small epsilon to avoid ties or to break ties towards larger bins?
    # No, Best Fit is standard.
    #
    # Let's try "Worst Fit" for a moment? Worst Fit tends to spread items, which can be bad for small items.
    #
    # Let's try a "Hybrid" approach.
    # Score = -remaining + beta * (bins - item) * (item / bins)
    #
    # Actually, let's look at the observed change p63 which got 1.01.
    # It likely modified the priority function.
    #
    # Let's try a score that considers the ratio of item to bin capacity.
    # Placing a small item in a large bin is wasteful.
    # Placing a large item in a small bin (if it fits) is efficient.
    #
    # Score = - (remaining / bins)
    # This normalizes the remaining capacity by the bin size.
    #
    # Let's try: Score = -remaining / max(bins, 1e-9)
    #
    # Or, let's try to minimize the "expected waste" assuming future items are similar to current.
    #
    # Let's try a simple improvement on Best Fit:
    # Best Fit is -remaining.
    # What if we use -remaining^1.5 or -remaining^0.5?
    # -remaining^0.5 would be less aggressive in filling bins (more like Worst Fit).
    # -remaining^1.5 would be more aggressive (even more Best Fit).
    #
    # Let's try -remaining^1.5.
    
    # Initialize scores with -inf for infeasible bins
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # Calculate scores for feasible bins
    # Using a power of 1.5 to emphasize filling bins more tightly than linear Best Fit.
    scores[feasible_mask] = -(remaining[feasible_mask] ** 1.5)
    
    return scores
# EVOLVE-BLOCK-END
