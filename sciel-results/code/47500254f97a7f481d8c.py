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
        
    # Strategy: Stochastic Inventory Cost Model
    #
    # We model the remaining capacity in each bin as an "inventory" of space.
    # The "cost" of keeping this space is the expected opportunity cost:
    # the probability that this space will NOT be used by any future item.
    #
    # We assume future items follow a Weibull distribution.
    # For Weibull distribution with shape k and scale lambda:
    # P(X > x) = exp(-(x/lambda)^k)
    #
    # The "value" of having remaining capacity 'r' is the probability that a
    # future item fits in it, i.e., P(X <= r).
    # Conversely, the "waste" or "cost" is P(X > r) = exp(-(r/lambda)^k).
    #
    # We want to minimize the total cost.
    # Total Cost = Sum over all bins of (Opportunity Cost of Remaining Space)
    #             + Penalty * (Number of Open Bins)
    #
    # When placing item 'item' in bin 'b':
    # - The new remaining space is r_new = b - item.
    # - The opportunity cost of this new space is exp(-(r_new/lambda)^k).
    # - If r_new is very small, exp(0) = 1? No, if r_new is small, P(X > r_new) is high?
    #   Wait. If r_new is 0, P(X > 0) = 1. This implies high cost?
    #   Actually, if r_new is 0, the bin is closed. There is no "opportunity cost" of holding space because there is no space.
    #   The "waste" is the space that *cannot* be used.
    #   If r_new = 0, waste = 0 (perfect fit).
    #   If r_new = 100, waste = 100 (empty bin, all space wasted if not used).
    #   
    #   Let's refine the cost function.
    #   The "dead space" is the amount of capacity that is too small to be useful.
    #   Or, simpler: The cost is the expected unused capacity.
    #   E[Unused] = integral from r to infinity of P(X > x) dx? No.
    #   
    #   Let's use a simpler proxy for "Stochastic Waste":
    #   Waste(r) = r * P(X > r) ? No.
    #   
    #   Let's look at the "Value of Space".
    #   Value(r) = P(X <= r) = 1 - exp(-(r/lambda)^k).
    #   We want to MAXIMIZE the Value of the remaining space?
    #   No, we want to MINIMIZE the Waste.
    #   Waste(r) = r - E[min(X, r)]?
    #   
    #   Actually, a common heuristic in stochastic bin packing is to minimize
    #   the "expected remaining space" weighted by its probability of being unused.
    #   
    #   Let's use the Survival Function S(r) = P(X > r) = exp(-(r/lambda)^k) as a 
    #   "deadness" factor.
    #   If r is small, S(r) is close to 1 (high chance future item is too big).
    #   If r is large, S(r) is close to 0 (high chance future item fits).
    #   
    #   We want to avoid creating small gaps (high S(r)) if we can avoid it?
    #   Actually, Best Fit creates small gaps. Worst Fit creates large gaps.
    #   
    #   Let's combine:
    #   Score = - (Expected Waste)
    #   Expected Waste of a bin with remaining r is approximated by:
    #   W(r) = r * S(r)  ? 
    #   If r=0, W=0.
    #   If r=100, S(100) is small, so W is small? That doesn't make sense.
    #   
    #   Let's try: Score = - (r * (1 - S(r))) - Penalty * I(r > 0)
    #   This penalizes keeping space open.
    #   
    #   Let's try a different angle: **Minimizing the Probability of Fragmentation**.
    #   
    #   Let's estimate lambda and k from the current state?
    #   We don't have the item history.
    #   We can assume typical Weibull parameters for capacity 100.
    #   Common Weibull for bin packing: k=1.5, lambda=20?
    #   Mean = lambda * Gamma(1+1/k). If lambda=20, k=1.5, Mean ~ 20 * 0.9 = 18.
    #   
    #   Let's use a fixed, robust set of parameters:
    #   k = 1.5
    #   lambda = 15.0 (Scale)
    #   
    #   Survival S(r) = exp(-(r/lambda)^k)
    #   
    #   We define the "Cost" of a bin with remaining r as:
    #   Cost(r) = r * S(r) + 0.1 * (1 if r > 0 else 0)
    #   Why r * S(r)?
    #   If r is small, S(r) ~ 1, Cost ~ r. (Linear cost for small gaps)
    #   If r is large, S(r) ~ 0, Cost ~ 0. (Large gaps are "cheap" because they will likely be used)
    #   Wait, if Cost is low for large r, we prefer Large r (Worst Fit).
    #   If Cost is linear for small r, we prefer Small r (Best Fit) relative to medium?
    #   
    #   Let's check the derivative.
    #   f(r) = r * exp(-(r/L)^k)
    #   f'(r) = exp(-(r/L)^k) - r * (k/L^k) * r^(k-1) * exp(-(r/L)^k)
    #   f'(r) = exp(-(r/L)^k) * (1 - k * (r/L)^k)
    #   Critical point at (r/L)^k = 1/k => r = L * k^(-1/k).
    #   For k=1.5, k^(-1/1.5) = 1.5^(-0.66) ~ 0.74.
    #   r_crit = 15 * 0.74 ~ 11.
    #   
    #   So Cost(r) increases from 0 to r=11, then decreases.
    #   This means r=11 is the MOST expensive gap.
    #   r=0 is Cost 0.
    #   r=100 is Cost ~ 0.
    #   
    #   This function penalizes "medium-small" gaps (around 10-15) heavily,
    #   and rewards "tight" gaps (0) and "large" gaps (>20).
    #   This is a **Bimodal** cost function, favoring Best Fit (tight) and Worst Fit (large),
    #   while avoiding the "dead zone" of medium gaps that are too small for typical items
    #   but too large to be ignored.
    #   
    #   Additionally, we add a small penalty for opening new bins (r > 0) to encourage
    #   consolidation if possible, but the bimodal nature might override this.
    #   Actually, if r=0, Cost=0. If r=100, Cost~0.
    #   We need to distinguish between "Full Bin" (r=0) and "Empty Bin" (r=100).
    #   Full Bin is better (fewer open bins).
    #   So we add a penalty for r > 0.
    #   Cost(r) = r * S(r) + 0.5 * (r > 0)
    #   
    #   Let's verify:
    #   r=0: Cost = 0 + 0 = 0. (Best)
    #   r=1: S(1) ~ 1. Cost ~ 1 + 0.5 = 1.5.
    #   r=11: S(11) ~ exp(-(11/15)^1.5) ~ exp(-0.63) ~ 0.53. Cost ~ 11*0.53 + 0.5 = 6.3. (Worst)
    #   r=50: S(50) ~ exp(-(50/15)^1.5) ~ exp(-8.5) ~ 0.0002. Cost ~ 0 + 0.5 = 0.5.
    #   r=100: S(100) ~ 0. Cost ~ 0.5.
    #   
    #   So, r=0 (Full) is best.
    #   r=50-100 (Large) is second best (Cost 0.5).
    #   r=1 (Tiny) is third (Cost 1.5).
    #   r=11 (Medium) is worst (Cost 6.3).
    #   
    #   This suggests:
    #   1. Prefer to fill a bin completely (Best Fit).
    #   2. If cannot fill completely, prefer a bin that leaves a LARGE gap (Worst Fit)
    #      over one that leaves a TINY gap, because tiny gaps are "dead" and incur a linear cost,
    #      while large gaps are "live" and incur only the fixed open-bin penalty.
    #      Wait, Cost(1) = 1.5, Cost(50) = 0.5.
    #      So we prefer r=50 over r=1.
    #      This is a **Worst Fit** preference for non-zero gaps!
    #      But we prefer r=0 over r=50.
    #      
    #   This is a hybrid: **Best Fit if possible, else Worst Fit**.
    #   This is a known strong heuristic: "Best Fit Decreasing" is good, but "Worst Fit"
    #   is also good. This dynamic switch might outperform static ones.
    #
    #   Let's implement this.
    
    # Parameters
    k = 1.5
    lambda_scale = 15.0
    
    valid_remaining = bins[valid]
    new_rem = valid_remaining - item
    
    # Calculate Survival Function S(r)
    # Handle r=0 case to avoid 0^k issues (though 0^1.5 is 0, exp(0) is 1)
    # S(r) = exp(-(r/lambda)^k)
    ratio = new_rem / lambda_scale
    # Clip to avoid numerical issues if new_rem is negative (shouldn't happen due to valid mask)
    ratio = np.maximum(ratio, 0.0)
    S = np.exp(-(ratio ** k))
    
    # Cost = r * S(r) + Penalty * (r > 0)
    # Penalty for keeping a bin open
    open_penalty = 0.5
    is_open = (new_rem > 1e-9).astype(float)
    
    cost = new_rem * S + open_penalty * is_open
    
    # We want to MINIMIZE cost.
    # So Score = -cost.
    scores = -cost
    
    # Mask invalid bins
    scores_full = np.full_like(bins, -np.inf)
    scores_full[valid] = scores
    
    return scores_full
# EVOLVE-BLOCK-END
