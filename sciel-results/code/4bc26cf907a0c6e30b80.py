import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    rem = bins - item
    C = 100.0
    
    # Fragmentation-aware scoring:
    # We want to avoid creating small, unusable fragments.
    # If an item takes a moderate fraction of the remaining space, it often leaves a remainder
    # that is too small for most other items (fragmentation) but too large to be "empty".
    
    # 1. Base score: Strongly prefer bins where the remaining capacity after placement
    #    is either very small (near 0) or very large (near C).
    #    This is the "Bimodal" preference.
    #    Let r = rem. We want to maximize a score that is high when r is near 0 or C.
    #    A simple way is to use -r*(C-r) which is max at C/2 and 0 at 0, C.
    #    So we want to MINIMIZE r*(C-r).
    
    # However, simple Best Fit (minimize rem) or Worst Fit (maximize rem) can be improved
    # by penalizing "wasteful" placements.
    
    # Let's look at the ratio of the item to the bin's current capacity.
    # If item is small relative to bin, we might prefer to fill it up? No, that's First Fit Decreasing logic which we don't have.
    
    # Let's refine the bimodal score with a fragmentation penalty.
    # The "dead space" is the remaining capacity after placement.
    # If we place item in a bin with remaining capacity `rem`, the new remaining is `rem`.
    # The "waste" is how hard it is to use that `rem`.
    
    # Hypothesis: Bins with `rem` close to 0 are great (bin is full).
    # Bins with `rem` close to C are good (bin is mostly empty, flexible).
    # Bins with `rem` in the middle (e.g., 20-80) are bad if they create fragments.
    
    # Let's try a score that is:
    # - High for small rem (Best Fit)
    # - High for large rem (Worst Fit)
    # - Low for medium rem
    
    # The current program uses: -base**2 where base = rem * (C - rem).
    # This penalizes the middle strongly.
    
    # Improvement idea:
    # 1. The penalty for the middle should be sharper or smoother depending on the distribution.
    # 2. We should account for the absolute size of the item.
    
    # Let's try a score based on the "efficiency" of the placement.
    # Efficiency = item / bins (how much of the bin we are filling).
    # But we also care about the leftover.
    
    # Alternative approach from literature:
    # "Harmonic" or "Next Fit" variants.
    
    # Let's stick to the bimodal family but tune it.
    # The current score: - (rem * (100 - rem))**2 - 0.1 * rem
    # The -0.1 * rem term biases towards smaller rem (Best Fit).
    # For Weibull distributions (which are skewed, many small items, few large),
    # Best Fit is generally good because it packs small items tightly.
    # However, if we fill a bin too tightly, we might leave a tiny gap that is useless.
    # If we leave a large gap, we can use it for future items.
    
    # Let's try to soften the bimodal penalty and increase the Best Fit bias slightly,
    # or make the bimodal peak narrower around 0.
    
    # A narrow peak around 0 means: "If you can fit it with very little waste, do it.
    # If you can't fit it with little waste, prefer a bin that stays very open (Worst Fit)
    # rather than creating a mediocre fragment."
    
    # Current: base = rem * (C - rem). Max at C/2.
    # We want to penalize the middle.
    # Let's try a higher power to make the "valley" deeper, or a different shape.
    
    # Let's try: score = - (rem / C)**alpha * (1 - rem/C)**beta
    # If alpha=2, beta=2, it's the current shape (scaled).
    # If we increase alpha, we penalize small rem less? No.
    # Let's look at the shape of f(x) = x*(1-x).
    # It is 0 at 0, 1, and 0.25 at 0.5.
    
    # What if we use: score = - rem * (C - rem) * k
    # And add a stronger linear term for Best Fit?
    
    # Let's try to optimize the parameters.
    # The current score is 1.0204.
    # The parent was 1.0101.
    # The "fragmentation-aware" inspiration 3 had 0.9803 (which is WORSE, since lower is better? 
    # Wait. "Score: mean excess over the L1 lower bound in percent (lower is better)."
    # "The evaluator measures the raw value (lower is better)."
    # "search score is seed/value... 1.0 equals seed... above 1.0 is an improvement"
    # So if the raw value is lower, the search score (seed/value) is HIGHER.
    # So 1.0204 is BETTER than 1.0000.
    # Inspiration 3 had score 0.9803, which means it was WORSE than the seed.
    # Inspiration 1 had score 1.0101, which is better than seed but worse than current.
    
    # So the current bimodal + small Best Fit bias is working well.
    # How to improve?
    
    # Maybe the "middle" penalty is too strong? Or not strong enough?
    # If we make the middle penalty stronger, we force items into bins that are either
    # almost full or almost empty. This reduces fragmentation.
    
    # Let's try increasing the power from 2 to 3 or 4.
    # base = rem * (C - rem)
    # score = -base**3
    
    # Also, the linear bias -0.1 * rem might be too small or too large.
    # If we have a strong bimodal penalty, the linear term is mainly for tie-breaking
    # between "almost full" (rem ~ 0) and "almost empty" (rem ~ C).
    # Best Fit (rem ~ 0) is usually better for packing efficiency.
    # So we want to prefer rem ~ 0 over rem ~ C when both are "good".
    # The current -0.1 * rem does this (negative score for larger rem).
    
    # Let's try:
    # 1. Increase power to 3.
    # 2. Increase linear bias slightly to 0.2 to strongly prefer Best Fit among the "good" bins.
    
    rem = bins - item
    C = 100.0
    
    # Ensure rem is non-negative for scoring, handle infeasible bins
    # Infeasible bins get a very low score
    
    # Bimodal penalty: penalize rem in the middle
    # Using a polynomial that is 0 at 0 and C, and max at C/2
    # f(rem) = rem * (C - rem)
    # We want to MINIMIZE this, so we use -f(rem)**power
    
    # Let's try power=2 (current) vs power=3
    # And tune the linear bias
    
    base = rem * (C - rem)
    # Clip base to avoid numerical issues if rem > C (shouldn't happen if rem = bins - item and bins <= C)
    # But if item > bin, rem is negative.
    
    # Mask for feasible bins
    feasible = rem >= 0
    
    # Calculate score for feasible bins
    # We want to maximize score.
    # Score = - (penalty for middle) - (penalty for large rem, i.e. prefer small rem)
    
    # Let's try a smoother transition.
    # score = - (base / (C/2)**2)**2  # Normalized bimodal
    # This makes the penalty scale with C.
    
    # Let's stick to the absolute scale.
    # base ranges from 0 to 2500 (at C/2).
    # base**2 ranges from 0 to 6,250,000.
    # base**3 ranges from 0 to 15,625,000,000.
    
    # The linear term is 0.1 * rem, max 10.
    # The bimodal term is huge. So the linear term is negligible except when base is very small (rem near 0 or C).
    # When rem is near 0, base is small.
    # When rem is near C, base is small.
    
    # So the competition is between rem=0 and rem=C.
    # For rem=0, base=0. Score = -0 - 0 = 0.
    # For rem=C, base=0. Score = -0 - 0.1*C = -10.
    # So rem=0 is preferred. Good.
    
    # What if we increase the power?
    # It makes the "valley" deeper, so bins in the middle are even less attractive.
    # This might help if the current strategy is still picking some "medium" bins.
    
    # Let's try power=2.5 or 3.
    
    score = np.zeros_like(rem)
    
    # Calculate for feasible
    # Use a slightly stronger bimodal penalty
    base_feasible = rem[feasible] * (C - rem[feasible])
    
    # Try power 2.5
    bimodal_penalty = base_feasible ** 2.5
    
    # Linear bias: prefer smaller rem (Best Fit)
    linear_bias = 0.5 * rem[feasible]  # Increased from 0.1 to 0.5
    
    score_feasible = -bimodal_penalty - linear_bias
    
    score[feasible] = score_feasible
    score[~feasible] = -1e9
    
    return score
