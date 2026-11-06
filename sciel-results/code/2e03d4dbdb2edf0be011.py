"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Best-fit: prefer bins that leave the least remaining capacity.
    # Add a small preference for bins that are already more filled (to consolidate)
    # but primarily minimize the leftover space.
    
    # bins is an array of remaining capacities.
    # We want to maximize the score.
    # Best Fit Decreasing (BFD) heuristic essentially picks the bin with the smallest remaining capacity that is >= item.
    # The score should be high for that bin.
    # Let remaining = bins - item.
    # We want to minimize remaining. So score = -remaining works.
    
    # However, simple best-fit can be improved.
    # One common improvement is to add a small penalty for opening new bins if possible, 
    # but here all bins are already open (capacity > 0).
    
    # Let's try a slight modification: 
    # If there are ties in remaining capacity, prefer the one with larger total capacity? 
    # We don't have total capacity, only remaining.
    
    # Actually, for Weibull distributions, items can be small or large.
    # Best-fit is generally strong.
    
    # Let's look at the "inspiration" which was just the seed.
    # The task asks to improve the seed.
    
    # Alternative: Worst Fit Decreasing? No, usually worse.
    # Alternative: First Fit? No, bins are ordered by index usually, but we return scores.
    
    # Let's try a "Best Fit" with a slight bias towards bins that are not "too" full 
    # to avoid fragmentation? Or bias towards fuller bins to consolidate?
    
    # Current: score = -(bins - item)
    # This is exactly Best Fit.
    
    # Perhaps we can use a non-linear function?
    # e.g. score = -log(bins - item + epsilon)?
    # This would strongly prefer bins where item fits almost exactly.
    
    # Or maybe: score = - (bins - item) / bins ? (Relative waste)
    
    # Let's try relative waste:
    # If bin has 100 left, item 10 -> waste 90, rel waste 0.9
    # If bin has 20 left, item 10 -> waste 10, rel waste 0.5
    # Best fit prefers 20. Relative waste prefers 20.
    
    # What if we prefer bins where the item takes a large portion of the remaining space?
    # score = item / bins ?
    # Higher is better.
    # If bins=100, item=10 -> 0.1
    # If bins=20, item=10 -> 0.5
    # This also prefers tighter fit.
    
    # Let's try combining absolute and relative?
    
    # Actually, a known good heuristic for online bin packing is "Best Fit Decreasing" 
    # but items arrive in random order (Weibull).
    
    # Let's try: score = -(bins - item) + 0.1 * bins
    # This adds a small preference for larger bins (Worst Fit component) to preserve small bins for small items?
    # This is a "Best Fit with bias towards larger bins" or "Harmonic" like?
    
    # Let's test the hypothesis: 
    # Pure Best Fit (current) might fragment if it keeps filling small bins exactly.
    # But usually Best Fit is very strong.
    
    # Let's try a different approach: 
    # The L1 lower bound is sum(items) / capacity.
    # We want to minimize number of bins.
    
    # Let's try "Worst Fit" for a moment? 
    # score = bins - item ? No, that's -remaining.
    # Worst Fit puts item in bin with most remaining space.
    # score = bins - item (maximize remaining after placing? No, maximize remaining BEFORE placing? No.)
    # Worst Fit: choose bin with max remaining capacity.
    # Score = bins.
    
    # Let's try a mix.
    # score = -(bins - item) + alpha * bins
    # = -bins + item + alpha * bins
    # = item + (alpha - 1) * bins
    # If alpha < 1, it's still Best Fit dominated.
    # If alpha = 0.5, score = item - 0.5 * bins.
    
    # Let's try alpha = 0.1.
    # score = -(bins - item) + 0.1 * bins
    
    # Wait, the inspiration 1 score is 1.0, same as seed.
    # The observed changes show p5 is a tie.
    
    # Let's try a simple non-linear best fit.
    # score = - (bins - item) ** 0.5 ?
    
    # Or maybe we should consider the distribution. Weibull has a shape parameter.
    # Without knowing it, we can't adapt dynamically.
    
    # Let's stick to a robust heuristic.
    # "Best Fit" is the seed.
    # Let's try "First Fit Decreasing" equivalent? No, online.
    
    # What if we penalize bins that are "too empty"?
    # i.e. if bins - item is very large, the score is very low.
    # This is already done by - (bins - item).
    
    # Let's try: score = -log(bins - item + 1e-9)
    # This makes the difference between 10 and 11 remaining much less significant than between 1 and 2.
    # It might help avoid over-optimizing for exact fits which might not be possible.
    
    # Let's try this.
    
    remaining = bins - item
    # Filter out bins that can't fit? 
    # The contract says: "scores each bin with enough remaining capacity".
    # The evaluator likely handles the filtering or expects -inf for invalid bins?
    # "each arriving item goes to the highest-scoring bin".
    # If a bin doesn't fit, it shouldn't be chosen.
    # The seed returns -(bins - item). If bins < item, remaining is negative, score is positive.
    # This is DANGEROUS. If bins=10, item=20, remaining=-10, score=10.
    # If another bin has bins=100, item=20, remaining=80, score=-80.
    # The evaluator would pick the bin that DOESN'T fit!
    # Wait, the seed is valid and scores 1.0.
    # The contract says: "scores each bin with enough remaining capacity".
    # This implies the evaluator might only pass bins that have enough capacity?
    # OR the function must return -inf for bins that don't fit.
    # "np.ndarray of one score per bin with enough remaining capacity"
    # This phrasing is ambiguous. Does it mean "return an array where invalid bins are marked invalid"?
    # Or "the input bins array already contains only valid bins"?
    # Given "bins: np.ndarray", it's likely the full array of remaining capacities.
    # If the seed works, it implies either:
    # 1. The evaluator filters bins before calling priority.
    # 2. The evaluator handles negative scores/invalid bins correctly (unlikely if seed doesn't mask).
    # 3. The seed is actually broken but happens to score 1.0? No, 1.0 is baseline.
    
    # Let's assume the evaluator filters or the seed is safe because 
    # maybe the evaluator ignores bins where item > bins?
    # To be safe, I should mask invalid bins to -inf.
    
    valid = bins >= item
    scores = np.full_like(bins, -np.inf, dtype=float)
    if np.any(valid):
        # Hybrid Best-Fit / Worst-Fit
        # alpha = 0.2 -> 80% Best Fit, 20% Worst Fit
        # Score = (1-alpha)*(-(bins - item)) + alpha*(bins)
        #       = (1-alpha)*item - (1-alpha)*bins + alpha*bins
        #       = (1-alpha)*item - (1-2*alpha)*bins
        alpha = 0.2
        scores[valid] = (1 - alpha) * item - (1 - 2 * alpha) * bins[valid]
    
    return scores
# EVOLVE-BLOCK-END
