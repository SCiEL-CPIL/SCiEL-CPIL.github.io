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
        
    # Strategy: Dynamic Waste Minimization
    # 
    # 1. Estimate the "typical" item size.
    # In online bin packing, if the average remaining capacity is high, 
    # items are likely small (or we are early). If low, items are large 
    # (or we are late). 
    # Let's estimate a "threshold" for useful space.
    # A common heuristic is that space less than the average item size is "dead".
    # We can approximate the average item size by looking at the distribution 
    # of remaining capacities. 
    # If bins were empty, remaining = 100. As items are added, remaining drops.
    # The mean remaining capacity (mean_rem) is a proxy for how "full" the system is.
    # However, a more robust proxy for "typical item size" in a steady state 
    # is related to the variance or just a fixed fraction of capacity if we don't 
    # have history. 
    # 
    # Let's use the mean remaining capacity of ALL bins as a proxy for the 
    # "average slack" in the system.
    # Actually, a better proxy for "typical item size" is often the median 
    # of the *filled* portions, but we don't have that directly.
    # 
    # Let's assume the "critical gap size" is proportional to the mean item size.
    # We can estimate mean item size as (Total Capacity - Total Remaining) / Count Items?
    # We don't know Count Items.
    #
    # Alternative Approach: "Gap Value" Function
    # For each valid bin, the gap created is g = bins[i] - item.
    # We want to maximize the "utility" of g.
    # Utility(g) should be:
    # - Low if g is very small (dead space, likely < min item size)
    # - High if g is large (flexible space)
    #
    # We can model Utility(g) as a smooth step or a simple quadratic/linear blend.
    # Let's define a "dead zone" threshold T.
    # If g < T, utility is low (penalized).
    # If g > T, utility is high (rewarded).
    #
    # What is T?
    # If we assume items are Weibull distributed with capacity 100, 
    # typical items might be around 10-30.
    # Let's set T dynamically based on the current state.
    # T = 0.1 * Capacity = 10? 
    # Or T = mean(bins) / 2?
    #
    # Let's try a score that rewards large gaps but heavily penalizes small gaps.
    # Score = g - Penalty * max(0, T - g)
    # This encourages gaps that are either "very small" (tight fit, efficient) 
    # or "very large" (flexible), but avoids "medium-small" gaps that are 
    # likely dead but not tight enough to be ideal.
    # Wait, usually Best Fit (small g) is good because it saves space.
    # Worst Fit (large g) is good because it keeps options open.
    # The "Dead Space" is the gap that is too small for future items but 
    # wastes space.
    #
    # Let's refine:
    # We want to minimize the number of bins.
    # Best Fit minimizes waste per bin.
    # Worst Fit reduces fragmentation.
    #
    # Let's use a "Harmonic Fit" or "Weighted Fit" approach.
    # Score = 1 / (1 + g)  -> Favors small g (Best Fit)
    # Score = g            -> Favors large g (Worst Fit)
    #
    # Let's try a strategy that adapts to the "fullness" of the bins.
    # If most bins are nearly full, we should Best Fit to avoid opening new bins.
    # If most bins are nearly empty, we should Worst Fit to spread load?
    # No, if bins are empty, it doesn't matter much.
    #
    # Let's look at the "density" of the bins.
    # mean_fill = 1 - mean(bins)/100
    #
    # New Idea: **Entropy-Based Scoring**
    # We want to keep the distribution of remaining capacities "healthy".
    # A healthy distribution might have low entropy (concentrated) or high entropy (spread)?
    # Usually, we want to avoid having many bins with small, unusable remainders.
    #
    # Let's try **Minimizing Expected Future Waste**.
    # Assume future items follow the same distribution as current items.
    # We can estimate the distribution of item sizes from the remaining capacities?
    # No, that's hard.
    #
    # Let's stick to a robust heuristic: **Adaptive Best Fit with Dead Space Penalty**.
    # 
    # 1. Calculate the gap for each valid bin: g = bins - item.
    # 2. Estimate a "dead space" threshold T.
    #    Let T = 5 (arbitrary small value) or T = 0.05 * 100.
    #    Actually, let's make T adaptive. 
    #    T = np.percentile(bins, 10) ? No.
    #    Let's use a fixed small threshold relative to capacity, say 5%.
    #    T = 5.0
    #
    # 3. Score = -g  (Best Fit)
    #    BUT, if g < T, we apply a heavy penalty to avoid creating tiny slivers.
    #    Why? Because if g is tiny, it's effectively dead space, and using a bin 
    #    that leaves a tiny sliver is worse than using a bin that leaves a 
    #    slightly larger gap that might be usable? 
    #    Actually, Best Fit is usually best because it minimizes total waste.
    #    The "dead space" argument is that if g is between 0 and T, it's dead.
    #    If g is between T and 2T, it might be dead or alive.
    #
    # Let's try a different structural change: **Rank-Based Hybrid**.
    # 
    # Calculate the rank of the gap g among all valid gaps.
    # If the system is "tight" (mean remaining < 50), use Best Fit (smallest gap).
    # If the system is "loose" (mean remaining > 50), use Worst Fit (largest gap).
    # 
    # This is similar to the current program's "Phase-Adaptive Fit", but 
    # let's change the metric.
    #
    # Current program uses fraction_large (bins > 50).
    # Let's use the **Mean Remaining Capacity** directly as a weight.
    # 
    # Weight w = mean(bins) / 100.0
    # If w is high (loose), we want Worst Fit (score = g).
    # If w is low (tight), we want Best Fit (score = -g).
    # 
    # Interpolate:
    # score = (2*w - 1) * g
    # If w > 0.5, weight is positive -> Worst Fit.
    # If w < 0.5, weight is negative -> Best Fit.
    #
    # This is structurally similar to the current one but uses mean instead of fraction.
    # Is that materially different? Not really.
    #
    # Let's try **Simulated Annealing / Stochastic Selection**?
    # No, must be deterministic for reproducibility in FunSearch? 
    # Actually, FunSearch allows randomness, but it's risky.
    #
    # Let's try **Lookahead with Distribution Assumption**.
    # Assume items are Weibull. 
    # We can estimate the scale parameter lambda from the current bins?
    # No.
    #
    # Let's try **Minimizing Variance of Remaining Capacities**.
    # Placing an item in a bin changes its remaining capacity.
    # We want to minimize the variance of the remaining capacities of ALL bins?
    # Or maximize it?
    # If we minimize variance, we make bins more similar.
    # If we maximize variance, we create some full bins and some empty bins.
    # Usually, creating full bins is good (they are "done").
    # So, we want to push bins towards 0 or 100?
    # 
    # Let's score based on how close the new remaining capacity is to 0 or 100.
    # new_rem = bins - item.
    # Distance to 0: new_rem
    # Distance to 100: 100 - new_rem
    # 
    # We want to minimize the "distance to full" if the bin is getting full,
    # or maximize the "distance to empty" if the bin is staying empty?
    #
    # Let's try: **Score = - (new_rem - 50)^2**
    # This prefers bins where the new remaining capacity is close to 50?
    # No, that's "Half Fit".
    #
    # Let's try: **Score = - abs(new_rem - 50)**
    # This prefers bins that are either very full or very empty after placement.
    # This encourages "polarization" of bins.
    # Bins become either full (0) or empty (100).
    # This is a known heuristic: "Best Fit Decreasing" tends to fill bins.
    # "Worst Fit" tends to keep bins empty.
    # Polarization might be good because it reduces the number of "half-full" 
    # bins that are hard to pack.
    #
    # Let's implement **Polarization Score**:
    # new_rem = bins - item
    # score = - (new_rem - 50)**2
    # This gives highest score (closest to 0) when new_rem is 50.
    # Wait, - (x-50)^2 is maximized at x=50.
    # So it prefers bins that become half-full.
    # That's probably bad.
    #
    # We want to prefer x close to 0 OR x close to 100.
    # So score = (new_rem - 50)**2 ?
    # Maximized at 0 or 100.
    # So we prefer bins that become very full or very empty.
    # 
    # Let's test this hypothesis.
    # If we put a small item in a full bin (new_rem ~ 0), score is high.
    # If we put a small item in an empty bin (new_rem ~ 100), score is high.
    # If we put a small item in a half-full bin (new_rem ~ 50), score is low.
    # This effectively says: "Use the bin that is already full, or the bin that is empty. Avoid the half-full ones."
    # This is a **Bimodal Preference**.
    
    valid_remaining = bins[valid]
    new_rem = valid_remaining - item
    
    # Score: Distance from 50.
    # We want to maximize this distance.
    scores = (new_rem - 50.0) ** 2
    
    # Mask invalid bins
    scores_full = np.full_like(bins, -np.inf)
    scores_full[valid] = scores
    
    return scores_full
# EVOLVE-BLOCK-END
