import time
from fractions import Fraction
from math import isqrt
import heapq

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s margin out of 90s limit
    
    # Known solution: [8, 31, 248]
    known = [8, 31, 248]
    best = known
    best_max = 248
    
    # Strategy: For each candidate max denominator M (from 247 down),
    # check if 5/31 can be written as sum of distinct unit fractions
    # all <= M, with at least one equal to M.
    # 
    # Use a BFS/DFS hybrid: from target fraction 5/31, repeatedly
    # subtract unit fractions 1/d where d <= M, tracking remaining.
    # 
    # Key optimization: use the fact that if we're looking for a solution
    # with max denominator exactly M, we can assume M is one of the terms.
    # Then we need to find k-1 other distinct denominators in [1, M-1]
    # that sum to 5/31 - 1/M.
    
    # For each M from 247 down to some lower bound:
    #   R = 5/31 - 1/M
    #   If R <= 0, skip (can't have positive unit fractions sum to non-positive)
    #   Find if R can be expressed as sum of distinct unit fractions with denominators in [1, M-1]
    
    # To make this tractable, use a meet-in-the-middle or iterative deepening approach.
    # 
    # Alternative: use the fact that for small M, we can enumerate subsets.
    # But 2^247 is infeasible.
    #
    # Better: use a greedy-like search with backtracking, but with strong pruning.
    #
    # Key pruning: 
    # 1. All denominators must be distinct
    # 2. All denominators <= M
    # 3. The sum must be exactly R
    #
    # Use the following approach:
    # - Sort candidate denominators in decreasing order of 1/d (i.e., increasing d)
    # - Use a priority queue to explore partial sums
    # - Track the set of used denominators to ensure distinctness
    
    # Actually, let me use a different approach: 
    # For each M, try to find a representation of R = 5/31 - 1/M
    # using at most 256 terms, all with denominators in [1, M-1].
    #
    # Use a recursive search with memoization on (remaining_fraction, min_denom, max_terms_left)
    # but this state space is too large.
    #
    # Instead, use the following key insight:
    # If R = p/q in lowest terms, then any unit fraction 1/d that we subtract
    # must satisfy that d divides something related to q.
    #
    # More precisely: if R = p/q, and we subtract 1/d, the new remainder is
    # p/q - 1/d = (pd - q)/(qd).
    # For this to be expressible as a sum of unit fractions with denominators <= M,
    # we need strong constraints.
    #
    # Let me try a different structural approach: use integer linear programming
    # or a SAT-like search over which denominators to include.
    #
    # Given the time constraints, let me use a more pragmatic approach:
    # For each M from 247 down to, say, 100, do a bounded DFS with aggressive pruning.
    
    for M in range(247, 99, -1):
        if time.time() > deadline:
            break
        
        R = TARGET - Fraction(1, M)
        if R <= 0:
            continue
        
        # Try to find a representation of R as sum of distinct unit fractions
        # with denominators in [1, M-1]
        #
        # Use iterative deepening: try with 1 term, 2 terms, 3 terms, ...
        # up to some limit (say 10 terms, since more is unlikely for small M)
        
        found = False
        for k in range(1, 11):  # Try up to 10 additional terms
            if time.time() > deadline:
                break
            
            # Check if k terms can sum to R with denominators in [1, M-1]
            # Min possible sum: sum of 1/(M-1), 1/(M-2), ..., 1/(M-k)
            # Max possible sum: 1/1 + 1/2 + ... + 1/k
            # If R is outside [min_sum, max_sum], skip this k
            
            min_sum = sum(Fraction(1, M - 1 - i) for i in range(k))
            max_sum = sum(Fraction(1, i + 1) for i in range(k))
            
            if R < min_sum or R > max_sum:
                continue
            
            # DFS to find k distinct denominators in [1, M-1] summing to R
            res = _find_k_terms(R, 1, M - 1, k, deadline)
            if res is not None:
                found = True
                best = res + [M]
                best_max = M
                break
        
        if found:
            break
    
    return best

def _find_k_terms(remaining, min_denom, max_denom, k, deadline):
    """
    Find k distinct denominators in [min_denom, max_denom] that sum to 'remaining'.
    Returns a list of denominators or None.
    """
    if time.time() > deadline:
        return None
    
    if k == 0:
        return [] if remaining == 0 else None
    
    if k == 1:
        if remaining.numerator == 1:
            d = remaining.denominator
            if min_denom <= d <= max_denom:
                return [d]
        return None
    
    if k == 2:
        # 1/a + 1/b = remaining
        # a < b, both in [min_denom, max_denom]
        # 1/a = remaining - 1/b => a = 1/(remaining - 1/b)
        # Iterate b from min_denom+1 to max_denom
        for b in range(max_denom, min_denom, -1):
            if time.time() > deadline:
                return None
            rem = remaining - Fraction(1, b)
            if rem <= 0:
                continue
            if rem.numerator == 1:
                a = rem.denominator
                if min_denom <= a < b:
                    return [a, b]
        return None
    
    # For k >= 3, use DFS with pruning
    # Lower bound for first denominator: 1/d <= remaining => d >= 1/remaining
    d_min = max(min_denom, int(1 / remaining) + (0 if 1 % remaining.denominator == 0 else 1))
    # Upper bound: we need k-1 more terms from [d+1, max_denom]
    # Max sum of k-1 terms from [d+1, max_denom] is sum(1/(d+1+i) for i in range(k-1))
    # This must be >= remaining - 1/d
    
    for d in range(d_min, max_denom - k + 2):
        if time.time() > deadline:
            return None
        
        if Fraction(1, d) > remaining:
            continue
        
        rem2 = remaining - Fraction(1, d)
        if rem2 <= 0:
            continue
        
        # Check if rem2 can be formed by k-1 terms from [d+1, max_denom]
        # Min sum: sum(1/(max_denom - i) for i in range(k-1))
        min_sum_next = sum(Fraction(1, max_denom - i) for i in range(k - 1))
        if rem2 < min_sum_next:
            continue
        
        # Max sum: sum(1/(d+1+i) for i in range(k-1))
        max_sum_next = sum(Fraction(1, d + 1 + i) for i in range(k - 1))
        if rem2 > max_sum_next:
            break
        
        # Recurse
        res = _find_k_terms(rem2, d + 1, max_denom, k - 1, deadline)
        if res is not None:
            return [d] + res
    
    return None
