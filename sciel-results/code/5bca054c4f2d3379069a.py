"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Weibull parameters for the generated sequences (inferred from typical task setups)
# Usually Weibull is parameterized by scale (lambda) and shape (k).
# Common defaults in such tasks: lambda=1, k=1 (Exponential) or k=2 (Rayleigh).
# Given capacity 100 and 1000 items, if mean item size is around 10-20, 
# let's assume a Weibull distribution that fits typical "online bin packing" benchmarks.
# A common setup is Weibull(k=2, lambda=sqrt(2*mean^2)) or similar.
# Without exact parameters, we assume a distribution where small items are more frequent.
# Let's assume a Weibull with k=1.5 and scale such that mean is ~15.
# Mean of Weibull(k, lambda) = lambda * Gamma(1 + 1/k).
# If k=1.5, Gamma(1+1/1.5) = Gamma(1.666) ~ 0.9027.
# If mean ~ 15, lambda ~ 15 / 0.9027 ~ 16.6.

# However, a more robust approach for the "value function" is to pre-compute
# the optimal greedy policy for small bins (e.g., capacity 0-20) using DP or simulation
# and then generalize. But we don't have the item distribution explicitly.
# 
# Alternative Strategy: "Adaptive Best Fit with Fragmentation Penalty"
# The key insight from the inspirations is that creating small fragments is bad.
# We can estimate the "waste" of a bin remainder by how likely it is to be filled by future items.
# If the remainder R is small, it's likely to become permanent waste.
# If the remainder R is large, it's likely to be filled.
# 
# Let's use a heuristic that penalizes remainders that fall into "dead zones".
# For Weibull distributions with k > 1, the PDF increases then decreases.
# For k < 1, it decreases.
#
# Let's try a different structural approach: **Lookahead Simulation**.
# Since we have a time budget, we can afford a light lookahead.
# But the function `priority` is called per item. We cannot do heavy simulation per call.
#
# Let's go back to the "Combinatorial Auction" / "Value Function" idea.
# We approximate the value of a bin with remaining capacity R after placing item x.
# V(R) = Expected future utility of this bin.
# If R is 0, V(0) = 0 (bin closed).
# If R is large, V(R) is high because it can fit many more items.
# 
# A simple approximation for V(R) is the expected number of items that can fit into R.
# E[N] = R / E[Item].
# This leads to First Fit Decreasing-like behavior (prefer large remainders).
# But Best Fit (prefer small remainders) is often better for packing density.
#
# The tension is between:
# 1. Filling a bin completely (Best Fit) to close it.
# 2. Keeping bins open with large capacity to accommodate large items.
#
# Strategy: **Hybrid with Weibull-aware Threshold**.
# Calculate the probability that a future item will fit into the remainder R.
# P(Next Item <= R) = CDF_Weibull(R).
# If P is high, the remainder is "valuable" because it will likely be used.
# If P is low, the remainder is "waste" because it will likely sit empty.
# 
# We want to minimize waste. So we prefer bins where the *new* remainder R' has a high probability of being used,
# OR where the current bin is almost full (R' is small, so P(Next <= R') is low, but we are closing the bin soon).
# 
# Actually, if we place item x in bin with remaining r, new remainder r' = r - x.
# If r' is small, it's hard to fill. This is "waste" unless the bin is closed.
# If r' is large, it's easy to fill. This is "good".
# 
# Wait, Best Fit works by minimizing r'. This creates many small remainders.
# Why does Best Fit work well? Because it packs items tightly, reducing the *number* of bins.
# The "waste" in Best Fit is the sum of small remainders.
# The "waste" in First Fit is the sum of large remainders in open bins.
# 
# For online packing, the goal is to minimize the number of bins.
# 
# Let's try a **Dynamic Programming-inspired Heuristic**.
# Pre-compute a "score table" for remainders 0..100 (capacity).
# The score S(r) represents how "good" it is to have a remainder r.
# We want to maximize the sum of S(r_i) for all open bins? No, we just pick the best bin.
# 
# Let's define the "Cost" of a bin with remainder r as the expected waste.
# Cost(r) = r * P(Next Item > r).
# This is the expected value of the remainder that will *not* be used by the next item.
# If the next item doesn't fit, the remainder is wasted (for now).
# 
# We want to place item x in bin b to minimize the *new* cost.
# New remainder r' = bins[b] - x.
# Cost(b) = r' * P(Next Item > r').
# 
# We need the CDF of the Weibull distribution.
# We don't know the exact parameters, but we can estimate them from the first few items?
# No, the function is stateless.
# 
# Let's assume a standard Weibull for this task. 
# Common in these benchmarks: Weibull(k=2, lambda=10) or similar.
# If we don't know, we can use a generic "exponential-like" decay for the survival function.
# S(r) = exp(-r/theta).
# 
# Let's refine the cost function:
# Cost(r') = r' * S(r').
# We want to minimize this cost.
# 
# Let's test this logic.
# If r' is very small, S(r') ~ 1, Cost ~ r'. (Small cost)
# If r' is very large, S(r') ~ 0, Cost ~ 0. (Small cost)
# If r' is medium, S(r') is significant, Cost is high.
# 
# This suggests we should prefer either very small remainders (Best Fit) or very large remainders (Worst Fit).
# This is a "Bimodal" preference.
# 
# Let's implement this with a tunable theta.
# We can estimate theta from the item size distribution if we had history, but we don't.
# Let's assume theta = 10.0 (arbitrary, but reasonable for capacity 100).
# 
# To make it more robust, we can use the item size itself to adjust.
# If item x is large, we are likely in a regime where large items are common.
# 
# Let's try: Score = - (r' * exp(-r'/theta)).
# We pick the highest score (lowest cost).
# 
# Let's add a time guard.
# 
# Wait, the prompt says "Weibull sequences". 
# If we assume Weibull(k=2, lambda=L), S(r) = exp(-(r/L)^2).
# Cost(r') = r' * exp(-(r'/L)^2).
# This function has a peak at r' = L.
# So we want to avoid remainders around L.
# We prefer remainders much smaller or much larger than L.
# 
# Let's assume L=10.
# 
# Implementation:
# 1. Calculate feasible bins.
# 2. For each feasible bin, calculate r' = bins[i] - item.
# 3. Calculate cost = r' * exp(-(r'/10.0)**2).
# 4. Score = -cost.
# 5. Return scores.
# 
# Let's refine L. If the mean item size is ~15, and Weibull k=2, 
# Mean = L * Gamma(1.5) = L * 0.886.
# L = 15 / 0.886 ~ 17.
# Let's use L=15.

import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 110.0  # Safety margin before 120s limit

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Check time limit
    if time.time() - _start_time > _TIMEOUT:
        # Fallback to simple Best Fit if running out of time
        remaining = bins - item
        scores = np.full_like(bins, -np.inf, dtype=float)
        feasible = remaining >= 0
        scores[feasible] = -remaining[feasible]
        return scores

    remaining = bins - item
    scores = np.full_like(bins, -np.inf, dtype=float)
    feasible = remaining >= 0
    
    if not np.any(feasible):
        return scores
    
    r_prime = remaining[feasible]
    
    # Weibull scale parameter estimate
    # Assuming items are Weibull(k=2, lambda=15)
    # Survival function S(r) = exp(-(r/lambda)^2)
    # Cost(r) = r * S(r)
    # We want to minimize Cost.
    
    lambda_est = 15.0
    # Use a vectorized calculation
    # To avoid overflow/underflow issues, we can use log domain or just direct exp
    # r is at most 100. (100/15)^2 = 44. exp(-44) is very small.
    
    cost = r_prime * np.exp(-(r_prime / lambda_est) ** 2)
    
    # We want to minimize cost, so we negate it for the score
    scores[feasible] = -cost
    
    return scores
# EVOLVE-BLOCK-END
