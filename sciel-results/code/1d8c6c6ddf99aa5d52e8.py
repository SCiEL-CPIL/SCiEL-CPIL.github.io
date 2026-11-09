# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track if we are in a "dynamic" mode if the evaluator allows it,
# but primarily we will use a highly refined static heuristic.
# The key insight from inspirations is that a simple balance heuristic is suboptimal.
# We need a hierarchy of features.
# Feature 1: Number of 1s. In cap sets, 1s are critical. 
# Vectors with 0 ones are in {0,2}^n (size 2^n).
# Vectors with 1 one can often be added.
# Vectors with many 1s conflict more.
# Let's try a score that peaks at a specific number of 1s or balances the "potential" for conflict.
# A known good strategy for small n is to prioritize vectors with few 1s, but not zero.
# However, the "balanced" inspiration (score 1.223) suggested closeness to mean.
# Let's refine the "balance" concept.
# Score = w1 * (count_1) + w2 * (count_0) + w3 * (count_2) + w4 * (variance_penalty)
# Actually, let's look at the "geometric-series tie-breaker" hint.
# This suggests the primary score is coarse, and the tie-breaker is fine-grained.
# Primary: Closeness to a "target" number of 1s.
# For n=8, max cap set is ~? 
# Let's try to prioritize vectors with exactly 1 or 2 ones, as they are "sparse" and conflict less.
# Wait, {0,2}^n has 0 ones. Size 2^n.
# If we add a vector with 1 one, say (1,0,0,0,0,0,0,0), it conflicts with (0,0,0,0,0,0,0,0) and (2,0,0,0,0,0,0,0)?
# AP: (0,0..0), (1,0..0), (2,0..0). Yes.
# So adding (1,0..0) forces us to drop one of the endpoints if both are present.
# But we gain 1 vector and lose 0 or 1?
# If we drop (2,0..0), net change 0.
# However, (1,0..0) might not conflict with other vectors in the set.
# The {0,2}^n set is "dense" in the 0/2 subspace.
# Vectors with 1s live in the "middle".
# A better heuristic might be to prioritize vectors with an *odd* number of 1s? Or specific patterns?
# Let's try a score based on the "weight" of the vector in a finite field sense.
# In F_3^n, a cap set is a set with no 3 points in a line.
# Lines are of the form {a, a+d, a+2d}.
# A vector v with many 1s is "central".
# Let's use a score that favors vectors with a specific count of 1s, say k, 
# where k is around n/3 or n/2.
# The previous "balanced" score was -abs(sum - n). Sum = 2*c2 + 1*c1. 
# If c0+c1+c2=n, sum = 2(n-c0-c1) + c1 = 2n - 2c0 - c1.
# Minimizing abs(sum-n) is maximizing closeness to n.
# n = 2n - 2c0 - c1 => 2c0 + c1 = n.
# This is a linear constraint on the counts.
# Let's try a different "balance": Minimize the sum of squares of digits?
# Or maximize the product of counts?
# Let's try: Score = - ( (c1 - n/3)^2 + (c0 - n/3)^2 + (c2 - n/3)^2 )
# This favors vectors where 0s, 1s, and 2s are equally distributed.
# For n=3, (1,1,1) has c1=3, c0=0, c2=0. Dist to (1,1,1) is 0? No, target is (1,1,1) for counts?
# If n=3, target counts are (1,1,1). 
# (1,1,1) has counts (0,3,0). Dist^2 = 1+4+1=6.
# (0,1,2) has counts (1,1,1). Dist^2 = 0. Score 0.
# (0,0,0) has counts (3,0,0). Dist^2 = 4+1+1=6.
# So this heuristic prioritizes permutations of (0,1,2).
# Are permutations of (0,1,2) good for cap sets?
# In n=3, {0,1,2} permutations are 6 vectors.
# Plus {0,2}^3 has 8 vectors.
# Can we combine them?
# (0,1,2) and (0,0,0) and (0,2,4->1)? No.
# Let's stick to the "balanced counts" heuristic but refine it.
# We will compute the count of 0s, 1s, 2s.
# Target: n/3 of each.
# Score = - sum((count_i - n/3)^2)
# This is a smooth surrogate for "balanced distribution".

def priority(vector, n):
    # Convert to numpy array for fast counting if needed, but vector is small.
    # vector is a tuple of ints.
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
            
    target = n / 3.0
    
    # Quadratic penalty for deviation from balanced distribution
    dist_sq = (c0 - target)**2 + (c1 - target)**2 + (c2 - target)**2
    
    # We want to maximize priority, so we negate the penalty
    base_score = -dist_sq
    
    # Tie-breaker: Use a geometric series on the digits to create a unique, fine-grained order.
    # This helps if many vectors have the same "balance".
    # Sum of (digit * 3^i) ? No, that's just the number value.
    # Let's use a hash-like function or a weighted sum with irrational weights to avoid collisions.
    # Or simply the sum of digits, then product of digits?
    # Let's use the sum of digits as a secondary tie-breaker.
    # Vectors with same balance might have different sums?
    # If counts are same, sum is same.
    # So we need a tertiary tie-breaker.
    # Use the "lexicographic" order of the vector itself?
    # To make it deterministic and spread out, we can use a simple hash.
    # Python's hash is randomized per run, which is bad for reproducibility if the evaluator relies on it?
    # But the evaluator just sorts by float.
    # Let's use a stable hash based on the vector content.
    # Sum of (v_i * (i+1)) ?
    tie_breaker = 0.0
    for i, v in enumerate(vector):
        tie_breaker += v * (i + 1)
        
    # Combine: Base score is dominant. Tie-breaker is small.
    # Scale tie-breaker to be much smaller than the difference in base scores.
    # Max dist_sq difference is roughly n.
    # Tie-breaker range is roughly n^2/2.
    # So divide tie_breaker by a large factor, e.g., 10000.
    
    final_score = base_score + tie_breaker / 10000.0
    
    return final_score
# EVOLVE-BLOCK-END
