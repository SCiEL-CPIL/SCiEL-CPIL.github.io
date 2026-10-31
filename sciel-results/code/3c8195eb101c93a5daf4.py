"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0  # safety margin before 150s kill
_START_TIME = time.time()

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline guard: if we're too close to the limit, return a safe default
    if time.time() - _START_TIME > _DEADLINE:
        valid = bins >= item
        if not np.any(valid):
            return np.full_like(bins, -np.inf)
        # Fallback: Best Fit (minimize remaining)
        scores = -(bins - item)
        scores[~valid] = -np.inf
        return scores

    # Determine which bins have enough remaining capacity
    valid = bins >= item
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
        
    # Strategy: Entropy-Based Scoring
    # 
    # Rationale:
    # In online bin packing, fragmentation often arises when bins get stuck at
    # similar intermediate fill levels, creating a "cluster" of bins that can
    # only accept very specific small items. 
    # 
    # We model the open bins (bins with remaining capacity > 0) as a discrete
    # probability distribution over capacity levels. The "health" of the packing
    # is related to the entropy of this distribution. High entropy implies a
    # diverse set of bin sizes, which is generally more robust to future random
    # item arrivals than a low-entropy (concentrated) set.
    #
    # However, calculating exact entropy change for every bin is expensive (O(N^2)
    # if we bin the capacities). We can approximate the impact on entropy by
    # looking at the "rarity" of the resulting state.
    #
    # Alternative Entropy Proxy: 
    # Instead of full Shannon entropy, we use the **Kullback-Leibler Divergence** 
    # intuition: we want to keep the distribution of remaining capacities 
    # "uniform" or "broad". 
    #
    # A simpler, computationally cheap proxy for "maintaining diversity" is to
    # penalize moves that make the current bin's remaining capacity too similar
    # to the *median* or *mode* of the other bins.
    #
    # Let's refine this to a specific, fast calculation:
    # We want to maximize the "spread" of the remaining capacities.
    # Variance is a measure of spread. Maximizing the variance of the remaining
    # capacities after placement is a good proxy for maintaining a healthy,
    # un-fragmented state.
    #
    # Let R be the vector of remaining capacities of all valid bins.
    # Let r_i be the remaining capacity of bin i.
    # Let new_r_i = r_i - item.
    #
    # We want to choose i that maximizes Var([r_1, ..., r_i - item, ..., r_N]).
    #
    # Expanding Variance:
    # Var(X) = E[X^2] - (E[X])^2
    #
    # Let S = sum(r_j) for all valid bins j.
    # Let S2 = sum(r_j^2) for all valid bins j.
    # N = number of valid bins.
    #
    # If we pick bin i, the new sum S' = S - item.
    # The new sum of squares S2' = S2 - r_i^2 + (r_i - item)^2
    #                             = S2 - r_i^2 + r_i^2 - 2*r_i*item + item^2
    #                             = S2 - 2*r_i*item + item^2
    #
    # New Mean Mu' = S' / N
    # New Variance V' = (S2' / N) - (Mu')^2
    #
    # We want to maximize V'.
    # Since N and item are constant for all candidates, and S and S2 are constants:
    # V' is a function of r_i.
    #
    # Let's expand V' to see the dependence on r_i:
    # V' = (S2 - 2*r_i*item + item^2)/N - ((S - item)/N)^2
    #
    # The term -(2*r_i*item)/N is the only part that depends on r_i.
    # To MAXIMIZE V', we need to MINIMIZE the term (2*r_i*item)/N.
    # Since item and N are positive constants, we need to MINIMIZE r_i.
    #
    # Wait. Minimizing r_i means choosing the bin with the SMALLEST remaining capacity
    # that fits the item. This is exactly **Best Fit**.
    #
    # Does maximizing variance imply Best Fit?
    # Let's check the logic.
    # If we have bins [10, 20, 30] and item 5.
    # Case 1: Put in 10 -> [5, 20, 30]. Mean 18.33. Var: (5-18.3)^2 + ...
    # Case 2: Put in 30 -> [10, 20, 25]. Mean 18.33. Var: (10-18.3)^2 + ...
    # 
    # Generally, moving a value closer to the mean reduces variance.
    # Moving a value further from the mean increases variance.
    # 
    # If we reduce r_i (by placing item), we are changing the value at index i.
    # If r_i is large (close to max), reducing it might bring it closer to the mean, reducing variance.
    # If r_i is small (close to min), reducing it might push it further from the mean (if mean is higher), increasing variance?
    #
    # Actually, the derivative analysis above showed:
    # V' depends on r_i via -2*r_i*item/N.
    # So V' is maximized when r_i is MINIMIZED.
    # So "Maximize Variance" heuristic leads to **Best Fit**.
    #
    # But Best Fit is often suboptimal for Weibull distributions because it creates many small gaps that become dead space.
    # 
    # Let's look at **Minimizing Variance**.
    # Minimizing V' means MAXIMIZING r_i.
    # This leads to **Worst Fit**.
    # Worst Fit keeps bins empty or full, avoiding the "middle" where fragmentation happens?
    # Actually, Worst Fit tends to keep bins empty, which is good for large items but bad for small ones (opens too many bins).
    #
    # Let's try a different structural metric: **Harmonic Mean of Remaining Capacities**.
    # We want to preserve the "effective capacity" of the system.
    #
    # Let's go back to the **Entropy** idea but implement it more directly.
    # Entropy H = - sum(p_k * log(p_k)).
    # We want to maximize H.
    #
    # Approximation:
    # If we place an item in bin i, we remove r_i from the distribution and add r_i - item.
    # 
    # Let's use a simpler proxy: **Distance from the Mode**.
    # We want to avoid bins that are "typical". We want to use bins that are "unusual" 
    # (either very full or very empty) to maintain a broad distribution.
    #
    # Score = - |r_i - Median(R)|
    # This is similar to the polarization score in the current program (distance from 50), 
    # but adaptive (distance from median).
    #
    # Let's try **Minimizing the Expected Number of Bins Opened**.
    # This is hard to estimate online.
    #
    # Let's try **First Fit Decreasing (FFD) emulation**.
    # FFD sorts items and places in first bin that fits.
    # Since we can't sort future items, we can't do FFD.
    #
    # Let's try **Next Fit Decreasing (NFD) emulation**.
    # NFD keeps one active bin and fills it until it's full, then opens a new one.
    # This is equivalent to: Always pick the bin with the SMALLEST remaining capacity? 
    # No, NFD picks the *current* bin. If we treat all open bins as "current", 
    # we should pick the one that is "most active" or "least full"?
    #
    # Actually, a known good heuristic for online packing is **Best Fit** with a small 
    # penalty for very small gaps (to avoid dead space).
    #
    # Let's try **Weighted Best Fit**.
    # Score = -g * w(g)
    # Where g is the gap.
    # If g is very small, w(g) is small (don't care so much about perfect fit if it's tiny).
    # If g is medium, w(g) is large (strongly prefer small g).
    #
    # Let's try a completely different family: **Randomized Best Fit with Bias**.
    # But we need deterministic.
    #
    # Let's look at the **L1 Lower Bound**.
    # The score is excess over L1.
    # L1 is sum(ceil(item_i / C)) * C? No, L1 for bin packing is usually just sum(items)/C.
    # The evaluator uses "mean percent excess of bins used over the L1 lower bound".
    # This implies we want to minimize the number of bins.
    #
    # Let's try **Minimizing the Sum of Squares of Remaining Capacities**.
    # This is equivalent to minimizing Variance + Mean^2.
    # Minimizing Sum of Squares (S2') is equivalent to minimizing (S2 - 2*r_i*item + item^2).
    # This is equivalent to MAXIMIZING r_i.
    # So this leads to **Worst Fit**.
    #
    # Let's try **Maximizing the Sum of Remaining Capacities**.
    # This is constant (Total Capacity - Total Items Placed).
    #
    # Let's try **Minimizing the Maximum Gap**.
    # This is hard.
    #
    # Let's go with **Adaptive Weighted Best Fit**.
    # We combine Best Fit (minimize gap) with a penalty for creating "dead" gaps.
    # A gap g is "dead" if g < threshold.
    # 
    # Score = -g - Penalty * max(0, Threshold - g)
    #
    # If g < Threshold, Score = -g - P*(T - g) = -g - P*T + P*g = (P-1)g - P*T.
    # If P > 1, the slope is positive? No, we want to DISCOURAGE small g.
    # We want the score to be LOW for small g.
    # Best Fit gives high score (low -g) for small g.
    # We want to lower the score for small g.
    # So we add a penalty that increases as g decreases.
    #
    # Score = -g - P * max(0, T - g)
    # For g < T: Score = -g - P(T-g) = (P-1)g - PT.
    # If P=2, Score = g - 2T.
    # As g -> 0, Score -> -2T.
    # As g -> T, Score -> T - 2T = -T.
    # For g > T: Score = -g.
    # As g -> T, Score = -T.
    # As g -> 100, Score -> -100.
    #
    # So for g < T, the score is HIGHER than -g (if P>1, (P-1)g > -g? No.
    # Let's check:
    # g=1, T=5, P=2.
    # -g = -1.
    # Penalty term: -2 * (5-1) = -8.
    # Total: -9.
    # So the score is LOWER (worse) for small gaps. This is what we want.
    #
    # So the strategy is: Best Fit, but heavily penalize gaps that are smaller than a threshold T.
    # This encourages "Good Fit" (g slightly larger than T) over "Perfect Fit" (g ~ 0).
    #
    # What is T?
    # T should be related to the expected minimum item size.
    # For Weibull with capacity 100, min item is likely around 5-10.
    # Let's set T = 10.
    
    valid_remaining = bins[valid]
    gaps = valid_remaining - item
    
    # Threshold for dead space
    T = 10.0
    # Penalty strength
    P = 2.0
    
    # Calculate scores for valid bins
    # Score = -gap - P * max(0, T - gap)
    penalty = P * np.maximum(0, T - gaps)
    scores_valid = -gaps - penalty
    
    # Mask invalid bins
    scores_full = np.full_like(bins, -np.inf)
    scores_full[valid] = scores_valid
    
    return scores_full
# EVOLVE-BLOCK-END
