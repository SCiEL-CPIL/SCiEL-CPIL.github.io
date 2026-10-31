"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score is preferred.
    
    Strategy:
    - We model the incoming items as following a Weibull distribution.
    - For each valid bin, we calculate the remaining capacity after placing the current item.
    - We estimate the probability that this remaining space will be "wasted" (i.e., not filled by future items) 
      using the Weibull survival function (CDF complement) evaluated at the remainder.
    - We score bins by minimizing this expected waste probability.
    - This effectively creates a dynamic Best-Fit that adapts to the distribution of item sizes,
      avoiding remainders that are too large to be filled by typical small items but too small for large ones.
    """
    scores = np.full(len(bins), -1e18)
    
    # Calculate remaining capacity in each bin
    remaining = bins - item
    
    # Only consider bins where remaining >= 0
    valid_mask = remaining >= 0
    if not np.any(valid_mask):
        return scores
    
    valid_indices = np.where(valid_mask)[0]
    valid_remaining = remaining[valid_mask]
    
    # Weibull parameters estimation for Weibull sequences with capacity 100.
    # Typical Weibull shapes for packing problems are around k=1.5 to 2.5.
    # Scale lambda is estimated from the mean.
    # For a Weibull(k, lambda), mean = lambda * Gamma(1 + 1/k).
    # Assuming mean item size is around 10-20 for capacity 100 (typical for 1000 items to fill ~50-100 bins).
    # Let's assume mean = 15.
    # If k = 2.0, Gamma(1.5) = sqrt(pi)/2 approx 0.886. lambda = 15 / 0.886 approx 16.9.
    # If k = 1.5, Gamma(1 + 1/1.5) = Gamma(1.666) approx 0.903. lambda = 15 / 0.903 approx 16.6.
    
    # Let's use k = 2.0 and lambda = 17.0 as a robust default for this regime.
    k = 2.0
    lam = 17.0
    
    # The probability that a future item is LARGER than the remainder r is S(r) = exp(-(r/lam)^k).
    # If S(r) is high, it's likely the remainder will be wasted (no future item fits or fits poorly).
    # If S(r) is low, it's likely a future item will be smaller than r and can fit.
    # We want to minimize S(r). So score = -S(r).
    
    # However, we also need to handle the case where r is very small.
    # If r is very small, S(r) is close to 1 (high waste probability if we consider "exact fit" or "filling").
    # Wait, the survival function S(r) = P(X > r).
    # If r is small, P(X > r) is high. This means it is likely that a future item is LARGER than r.
    # This implies r is "dead" space because future items won't fit in it (they are too big).
    # If r is large, P(X > r) is low. This means it is likely that future items are SMALLER than r.
    # This implies r is "live" space because future items can fit in it.
    
    # So minimizing S(r) is correct: we prefer remainders r where it is unlikely that future items are larger than r.
    # i.e., we prefer remainders that are "large enough" to accept typical future items.
    
    # But wait. If r is HUGE (e.g., 90), S(90) is very small. Score is high.
    # If r is SMALL (e.g., 1), S(1) is close to 1. Score is low.
    # This suggests we should prefer bins that have LOTS of remaining space?
    # That sounds like Worst-Fit, which is often bad for bin packing (creates many partially filled bins).
    
    # Let's re-evaluate.
    # The goal is to minimize the number of bins.
    # Waste occurs when a bin is closed (no more items fit) and has empty space.
    # A remainder r is "waste" if no future item fits in it.
    # An item fits if X <= r.
    # Probability an item fits is CDF(r) = 1 - S(r).
    # Probability an item DOES NOT fit is S(r).
    
    # If we choose a bin with remainder r, the "risk" is that this space becomes waste.
    # But actually, the space is only waste if the bin is eventually closed.
    
    # Let's look at the "dead space" concept.
    # In bin packing, "dead space" is space that cannot be filled by any future item.
    # If items are i.i.d. from a distribution, a remainder r is "dead" if P(X <= r) is very low? No.
    # If P(X <= r) is low, it means most items are larger than r. So r is dead.
    # If P(X <= r) is high, it means most items are smaller than r. So r is live.
    
    # So we want to maximize P(X <= r) = CDF(r).
    # Or minimize S(r).
    
    # Why did Worst-Fit (maximizing r) fail?
    # Worst-Fit maximizes r, which minimizes S(r).
    # So my derivation suggests Worst-Fit is good?
    # But Worst-Fit is known to be inferior to Best-Fit in many cases.
    
    # The flaw in the simple survival argument:
    # It ignores the fact that if you put an item in a bin with large r, you are "consuming" a large bin.
    # If you put it in a bin with small r (but still fitting), you are "completing" a bin.
    
    # Let's try a different angle: The "Gap" distribution.
    # We want to avoid creating gaps that are "awkward".
    # Awkward gaps are those that are too small to fill with small items (because they are too small? No, small items fill small gaps) 
    # or too large to fill with large items?
    
    # Actually, Best-Fit is good because it minimizes the remaining space, making the bin "fuller".
    # A fuller bin is less likely to be opened again, and if it is, the remaining space is small, 
    # so it's less likely to waste a large chunk of capacity.
    
    # Let's look at the "Polarization" insight from the knowledge base:
    # "Polarization scoring (maximizing distance from half-capacity) ... preserving half-full bins for future medium items while consolidating small items"
    
    # This suggests that remainders near the "median" item size are valuable (can be filled by medium items).
    # Remainders near 0 are "good" (bin is full).
    # Remainders near Capacity are "bad" (bin is empty, wasted potential).
    
    # So we want to maximize the "fillability" of the remainder.
    # Fillability is high if r is close to typical item sizes.
    # Fillability is low if r is very small (no room for anything else) OR very large (too much space, likely to remain partially filled).
    
    # Wait, if r is very small, it's good because the bin is almost full.
    # If r is very large, it's bad because we are spreading items thin.
    
    # So we want to minimize r? That's Best-Fit.
    # But Polarization says "maximizing distance from half-capacity".
    # Distance from half-capacity (50) is |r - 50|.
    # Maximizing |r - 50| means preferring r close to 0 or r close to 100.
    # Preferring r close to 0 is Best-Fit.
    # Preferring r close to 100 is Worst-Fit.
    # How does it choose? "adapts between Best-Fit and Worst-Fit based on item size".
    
    # If item is small, we want to consolidate -> Best-Fit (r close to 0).
    # If item is large, we want to preserve space for other large items? -> Worst-Fit (r close to 100)?
    # Or maybe: if item is large, putting it in a nearly full bin (small r) is good because it completes the bin.
    # If item is small, putting it in a nearly full bin is good.
    
    # Let's stick to the Weibull survival idea but refine it.
    # The "waste" is not just P(X > r).
    # The waste is the expected empty space left when the bin is closed.
    
    # Let's use a simpler, robust heuristic that captures "fillability":
    # Score = -r + alpha * P(X <= r)
    # This combines Best-Fit (-r) with a bonus for "live" space (P(X <= r)).
    
    # If r is small, -r is high (good), P(X <= r) is low (bad).
    # If r is large, -r is low (bad), P(X <= r) is high (good).
    # The alpha balances these.
    
    # For small items, we want Best-Fit dominant.
    # For large items, we might want the P(X <= r) term to matter more?
    
    # Let's try:
    # score = -r + 0.1 * CDF(r)
    
    # CDF(r) = 1 - exp(-(r/lam)^k)
    
    valid_r = valid_remaining
    # Calculate CDF
    # Clip r to avoid numerical issues if r is very large (though max is 100)
    cdf = 1.0 - np.exp(-np.power(valid_r / lam, k))
    
    scores[valid_indices] = -valid_r + 0.1 * cdf
    
    return scores
# EVOLVE-BLOCK-END
