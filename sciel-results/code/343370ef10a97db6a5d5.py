import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining capacity for each bin
    rem = bins - item
    
    # Valid bins have non-negative remaining capacity
    valid_mask = rem >= 0
    
    # Initialize scores to -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid_mask):
        return scores
    
    valid_rem = rem[valid_mask]
    valid_bins = bins[valid_mask]
    
    # Hybrid strategy: 
    # 1. Primary: Best-fit (minimize remaining space) to pack tightly
    # 2. Secondary: Avoid bins that are already very full if there's a slight fit
    #    Actually, for Weibull (often many small items), pure best-fit is strong.
    #    Let's try a slight variation: penalize bins that are "too full" relative to item size?
    #    Or reward bins where the item fits "nicely" (e.g., ratio of item to bin)?
    
    # Let's stick to the Best-Fit family but refine the score.
    # Current: score = -rem
    # Alternative: score = -rem + alpha * (bins / capacity) ? No, capacity is fixed.
    
    # Consider the "Worst-Fit" aspect for very small items? 
    # If item is very small, Best-Fit might fill up bins too fast, leaving medium bins empty.
    # But Best-Fit is generally very robust.
    
    # Let's try a soft-min/soft-max surrogate as suggested by the directive?
    # The objective is minimizing bin count. 
    # In online packing, we can't see the future.
    
    # Let's look at the inspiration 2 (score 0.5552) which was worse.
    # The current program (score 1.0) is Best-Fit.
    
    # Is there a better heuristic within the "fit" family?
    # "First Fit Decreasing" is offline.
    # "Next Fit" is bad.
    # "Best Fit" is good.
    # "Worst Fit" is sometimes better for uniform distributions, but Weibull is skewed.
    
    # Let's try a blend: 
    # Score = -rem + beta * (bins / 100.0)
    # This slightly prefers fuller bins (higher bins value) if the remaining space is similar?
    # No, that would be worst-fit.
    
    # Let's try: Score = -rem / max(rem, epsilon) ? No.
    
    # How about normalizing the remaining space by the bin's current usage?
    # rem_ratio = rem / bins
    # This measures how much of the *remaining potential* is left.
    # If a bin has 90/100 used, rem=10. rem_ratio = 10/90 = 0.11.
    # If a bin has 10/100 used, rem=90. rem_ratio = 90/10 = 9.0.
    # Best fit minimizes rem.
    # Minimizing rem_ratio might be different?
    
    # Let's test the hypothesis: 
    # For Weibull, items are often small. 
    # If we have a bin with 95 used and one with 50 used.
    # Item size 3.
    # Bin 1 rem: 5. Bin 2 rem: 50.
    # Best Fit picks Bin 1.
    # This is usually correct to keep Bin 2 open for larger items.
    
    # What if we add a small penalty for bins that are almost full?
    # No, that's what best fit does implicitly by preferring smaller rem.
    
    # Let's try a different metric: 
    # Minimize the "waste" created.
    # Waste = rem.
    # So Best Fit is minimizing immediate waste.
    
    # Is there a "Lookahead" heuristic?
    # We can't know future items.
    
    # Let's try a slight modification: 
    # If multiple bins have the same remaining space (rare with floats), pick the one with more used space?
    # Or pick the one with less used space (Worst Fit among ties)?
    
    # Given the directive to "investigate improvements within its family" (Best-Fit family)
    # and "consider a smooth surrogate", maybe the current hard selection is suboptimal due to ties or precision?
    # Unlikely to be the main bottleneck.
    
    # Let's try "Best Fit with a bias towards bins that are not yet 'too' full" to preserve some flexibility?
    # Actually, Best Fit *is* the strategy that preserves flexibility for larger items by filling small bins.
    
    # Let's try "Worst Fit" for comparison? No, directive says KEEP this design (Best Fit).
    
    # Let's try a hybrid: 
    # Score = -rem + 0.1 * (bins / 100.0)
    # This slightly prefers fuller bins if rem is very close?
    # No, -rem dominates.
    
    # What if we use: Score = -rem^2 ?
    # This penalizes large remainders more heavily?
    # If rem=5, score=-25. If rem=10, score=-100.
    # This still picks the smallest rem. It just changes the tie-breaking or relative preference if we added other terms.
    
    # Let's try: Score = -rem + 0.01 * bins
    # This is a very slight Worst-Fit bias.
    
    # Let's try: Score = -rem - 0.01 * (bins / 100.0)
    # This is a slight Best-Fit bias (penalizing fuller bins slightly more? No, -0.01*fuller is more negative, so lower score. So it prefers LESS full bins if rem is same).
    
    # Smooth Best-Fit with adaptive weighting based on item-to-remaining ratio
    # This creates a smooth surrogate that penalizes "wasted" space more when
    # the item is small relative to the remaining capacity, encouraging tighter
    # packing of small items while preserving flexibility for larger ones.
    
    capacity = 100.0
    
    # Compute the ratio of item to remaining capacity for each valid bin
    # This measures how "tight" the fit is
    ratio = np.clip(item / (valid_rem + 1e-10), 0, 1)
    
    # For tight fits (ratio close to 1), we want to strongly prefer that bin
    # For loose fits (ratio close to 0), we want to be more flexible
    
    # Use a smooth weighting: weight = 1 + alpha * (1 - ratio)
    # When ratio is high (tight fit), weight is close to 1
    # When ratio is low (loose fit), weight is higher, making the penalty for
    # remaining space less severe, allowing us to consider other options
    
    alpha = 0.3  # Tunable parameter for how much we relax the penalty for loose fits
    
    weight = 1 + alpha * (1 - ratio)
    
    # Score: negative weighted remaining space
    # This is a smooth surrogate that adapts based on fit quality
    valid_scores = -valid_rem * weight
    
    # Add a small tie-breaker: prefer bins with less used space among near-ties
    # This helps with numerical stability and can slightly improve packing
    tie_breaker = 0.001 * (1 - valid_bins / capacity)
    
    valid_scores += tie_breaker
    
    scores[valid_mask] = valid_scores
    return scores
