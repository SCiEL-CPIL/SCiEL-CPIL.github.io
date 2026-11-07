import time
import math
import sys
from collections import defaultdict

def construct():
    deadline = time.time() + 55.0  # safety margin before 60s
    n = 210
    max_val = 10210
    
    # Precompute square-free kernels up to max_val
    # For each number, compute its square-free part
    # We need: product of selected numbers * 210 is a perfect square
    # This means the square-free part of (product * 210) must be 1
    # Equivalently, the XOR-sum (mod 2) of the prime exponents in each selected number plus 210 must be 0
    
    # Strategy: Find a set of distinct integers in (210, 10210] such that
    # their product * 210 is a perfect square
    # 
    # Key insight: We can use pairs. If a and b have the same square-free kernel k,
    # then a*b has square-free kernel 1 (since k*k = k^2 is a square).
    # So if we find a*b*210 = square, we need square-free(a*b) = square-free(210)
    # square-free(210) = 2*3*5*7 = 210
    # 
    # More generally: product of selected * 210 = square
    # => square-free(product of selected) = 210 (since sf(210)=210 and sf(a*b)=sf(sf(a)*sf(b)) mod squares)
    # Wait, that's not quite right. Let me think more carefully.
    #
    # Let P = product of selected numbers. We need P * 210 = s^2.
    # This means P * 210 is a perfect square.
    # Equivalently, for every prime p, the exponent of p in P*210 is even.
    #
    # Simplest approach: Find two numbers a, b such that a*b*210 is a perfect square.
    # This requires sf(a*b) = sf(210) = 210... no wait.
    # a*b*210 = s^2 means sf(a*b*210) = 1.
    # sf(a*b*210) = sf(sf(a)*sf(b)*210) where we cancel common factors.
    #
    # Let me just search for pairs directly.
    
    def is_square(m):
        if m < 0:
            return False
        s = math.isqrt(m)
        return s * s == m
    
    # Search for pairs (a, b) with 210 < a < b <= 10210 such that a*b*210 is a perfect square
    # To minimize max(a,b) - 210, we want small a and b.
    
    best_max = max_val + 1
    best_result = None
    
    # For a*b*210 to be a square, we need a*b = s^2 / 210 = s^2 * 210 / 210^2... 
    # Actually a*b*210 = k^2 => a*b = k^2/210. So 210 must divide k^2.
    # Since 210 = 2*3*5*7, k must be divisible by 2*3*5*7 = 210.
    # Let k = 210*t. Then a*b = (210*t)^2 / 210 = 210*t^2.
    # So a*b = 210*t^2.
    # We need a and b to be distinct integers in (210, 10210].
    # So we need to find factorizations of 210*t^2 into a*b where 210 < a < b <= 10210.
    
    # Iterate t from 1 upward, compute M = 210*t^2, find divisors a of M in range (210, min(M//2, 10210))
    # such that b = M//a is also in (210, 10210] and a != b.
    
    # We want to minimize max(a,b), so try small t first.
    
    t = 1
    while time.time() < deadline:
        M = 210 * t * t
        # a must divide M, a > 210, a < b = M/a, so a < sqrt(M)
        # Also b <= 10210, so a >= M/10210
        # And a <= 10210
        # a > 210
        lo = max(211, (M + 10209) // 10210)  # a >= M/10210, so b = M/a <= 10210
        hi = int(math.isqrt(M))  # a < b implies a < sqrt(M)
        if lo > hi:
            t += 1
            continue
        
        found = False
        for a in range(lo, hi + 1):
            if time.time() >= deadline:
                break
            if M % a != 0:
                continue
            b = M // a
            if b > 10210 or b <= a:
                continue
            if b > 210:
                candidate_max = b  # since b > a > 210, max is b
                if candidate_max < best_max:
                    best_max = candidate_max
                    best_result = (a, b)
                    found = True
                    # We found one for this t; since we're iterating a in order,
                    # b = M/a is decreasing as a increases, so the first valid a gives largest b.
                    # We want smallest b, so we should check all a or break early?
                    # Actually for fixed t, as a increases, b decreases. So the last valid a gives smallest b.
                    # Let's not break, continue to find smaller b.
        if found:
            # For this t, we found at least one pair. The best for this t has smallest b.
            # Since we want to minimize overall max, and larger t gives larger M thus likely larger values,
            # we can potentially stop if we've found a good solution.
            # But let's continue a bit more to see if we can do better.
            if best_max - 210 <= 10:
                break
        t += 1
        if t > 100000:  # safety
            break
    
    # Also try single element: a * 210 = s^2 => a = s^2/210. 
    # Need 210 | s^2, so 210 | s. s = 210k, a = 210k^2.
    # Need 210 < 210k^2 <= 10210 => 1 < k^2 <= 48.6 => k=2,3,4
    # k=2: a=840, check: 840*210 = 176400 = 420^2. Yes!
    # So [840] works with value 840-210=630.
    
    if best_result is None:
        # Fallback to single element
        return [840]
    
    # Verify
    a, b = best_result
    prod = a * b * 210
    if is_square(prod):
        return [int(a), int(b)]
    else:
        return [840]
