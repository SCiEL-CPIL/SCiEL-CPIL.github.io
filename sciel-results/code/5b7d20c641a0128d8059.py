"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    from heapq import heappush, heappop
    import math

    target = Fraction(5, 31)
    
    # We want to minimize the largest denominator in an Egyptian fraction expansion.
    # This is equivalent to finding a set of distinct integers d_1, ..., d_k such that
    # sum(1/d_i) = 5/31 and max(d_i) is minimized.
    #
    # We can model this as a search problem where we build the expansion term by term.
    # State: (remaining_fraction, last_denominator)
    # We use Dijkstra-like search where the cost is the largest denominator used so far.
    # However, since we want to minimize the MAX denominator, and we add terms one by one,
    # we can do a BFS/DFS with pruning based on the current max denominator.
    #
    # A better approach: Binary search on the maximum allowed denominator M.
    # For a given M, check if there exists an Egyptian fraction expansion of 5/31
    # using distinct denominators all <= M.
    #
    # The check can be done via DFS with memoization or backtracking.
    # Since the denominators must be distinct and we're summing reciprocals,
    # we can enforce an ordering: d_1 < d_2 < ... < d_k to avoid permutations.
    #
    # Let's use a recursive search with memoization.
    # State: (remaining_numerator, remaining_denominator, min_next_denominator)
    # But the state space might be large. Let's try a direct search with good pruning.
    #
    # Alternative: Since the target is 5/31 ≈ 0.161, and we want to minimize the max denominator,
    # the optimal solution likely has a small number of terms with moderate denominators.
    #
    # Let's try a BFS from the target, subtracting 1/k for increasing k, and track the max denominator.
    # We use Dijkstra where the priority is the current max denominator.
    
    # Dijkstra's algorithm
    # State: (Fraction remaining, int min_denominator_allowed, list of denominators used)
    # Actually, we don't need the full list for the state, just the remaining fraction and the constraint that
    # all future denominators must be > last_denominator (to ensure distinctness and ordering).
    # But for minimizing the max, we need to track the max so far.
    
    # Let's define the state as (remaining_fraction, last_denominator) where last_denominator is the 
    # smallest denominator we can use next (to ensure distinctness, we use denominators in increasing order).
    # The cost is the maximum denominator used so far. But we don't track that in the state.
    
    # Actually, let's think differently. We want to find min M such that there exists an expansion with all d_i <= M.
    # We can binary search M. For each M, we do a DFS to see if we can reach 0.
    
    # Let's implement a DFS with memoization for a fixed M.
    # State: (num, den, min_d) where remaining = num/den, and all future denominators must be >= min_d.
    # We also need to ensure distinctness, which is handled by min_d.
    
    from functools import lru_cache
    
    def can_expand(num, den, min_d, M):
        """Check if num/den can be expressed as sum of 1/d_i with d_i >= min_d and d_i <= M, all distinct."""
        if num == 0:
            return True
        if num < 0:
            return False
        # If num/den is too large to be covered by remaining terms
        # Max possible sum with denominators from min_d to M is sum(1/d for d in range(min_d, M+1))
        # But that's expensive. Let's just try to find a term.
        
        # The next denominator d must satisfy:
        # 1. d >= min_d
        # 2. d <= M
        # 3. 1/d <= num/den  =>  d >= den/num
        # 4. After subtracting 1/d, the remainder (num/den - 1/d) must be expressible with denominators > d.
        
        # d must be at least max(min_d, ceil(den/num))
        d_min = max(min_d, (den + num - 1) // num)
        if d_min > M:
            return False
        
        for d in range(d_min, M + 1):
            # Check if 1/d <= num/den
            if d * num > den:
                # 1/d > num/den, so we can't use this d
                # Actually, if d * num > den, then 1/d > num/den, so remainder would be negative
                # Wait: num/den - 1/d = (num*d - den) / (den*d)
                # We need num*d - den >= 0 for non-negative remainder
                # If num*d < den, remainder is negative, skip
                # If num*d == den, remainder is 0, success
                if d * num == den:
                    return True
                if d * num < den:
                    continue
            # Compute remainder: num/den - 1/d = (num*d - den) / (den*d)
            new_num = num * d - den
            new_den = den * d
            # Reduce fraction
            g = math.gcd(new_num, new_den)
            new_num //= g
            new_den //= g
            
            if new_num == 0:
                return True
            
            # Recurse with min_d = d + 1
            if can_expand(new_num, new_den, d + 1, M):
                return True
        
        return False
    
    # Binary search on M
    lo, hi = 1, 10000000
    best_M = None
    
    # We need to find the minimum M such that can_expand(5, 31, 1, M) is True.
    # But we also need to reconstruct the actual denominators.
    
    # Let's modify to also return the denominators.
    
    def find_expansion(num, den, min_d, M):
        """Find an expansion of num/den with denominators in [min_d, M], all distinct, in increasing order."""
        if num == 0:
            return []
        
        d_min = max(min_d, (den + num - 1) // num)
        if d_min > M:
            return None
        
        for d in range(d_min, M + 1):
            if d * num < den:
                continue
            if d * num == den:
                return [d]
            
            new_num = num * d - den
            new_den = den * d
            g = math.gcd(new_num, new_den)
            new_num //= g
            new_den //= g
            
            if new_num == 0:
                return [d]
            
            rest = find_expansion(new_num, new_den, d + 1, M)
            if rest is not None:
                return [d] + rest
        
        return None
    
    # Binary search for the smallest M
    lo, hi = 1, 10000000
    result = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if find_expansion(5, 31, 1, mid) is not None:
            result = find_expansion(5, 31, 1, mid)
            hi = mid - 1
        else:
            lo = mid + 1
    
    if result is not None:
        return result
    else:
        # Fallback to greedy
        r = Fraction(5, 31)
        values = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values

# EVOLVE-BLOCK-END
