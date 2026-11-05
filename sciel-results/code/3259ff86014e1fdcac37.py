"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import factorial, isqrt
    from functools import lru_cache

    target = factorial(12)  # 479001600

    # Generate all divisors of target
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divs = []
        for d in divisors:
            for k in range(e + 1):
                new_divs.append(d * (p ** k))
        divisors = new_divs
        
    # Filter divisors > 12 and sort
    valid_divs = sorted([d for d in divisors if d > 12])
    
    # Binary search on the maximum factor M
    # We want the smallest M such that there exists a set of distinct divisors
    # from (12, M] whose product is target.
    
    def can_form(max_val):
        """
        Returns True if target can be formed by distinct divisors d where 12 < d <= max_val.
        """
        # Filter candidates
        candidates = [d for d in valid_divs if d <= max_val]
        if not candidates:
            return False
        
        # To speed up DFS, we can use a set for O(1) lookup if we iterate divisors of remaining,
        # but iterating candidates is also fine if candidates list is small.
        # For small max_val, candidates list is small.
        
        # We need to find a subset of candidates with product target.
        # Use DFS with ordering to avoid permutations.
        # Sort candidates in ascending order.
        candidates.sort()
        
        # Memoization might help, but state is (index, remaining).
        # Index can be up to len(candidates). Remaining is a divisor of target.
        # Number of states: len(candidates) * num_divisors(target).
        # num_divisors(12!) = (10+1)*(5+1)*(2+1)*(1+1)*(1+1) = 11*6*3*2*2 = 792.
        # len(candidates) can be up to 791.
        # 791 * 792 ~ 626k states. This is very manageable.
        
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dfs(idx, remaining, count):
            if remaining == 1:
                return count > 0
            if count >= 12:
                return False
            if idx >= len(candidates):
                return False
            
            # Pruning: 
            # The smallest product we can form with (12-count) factors from candidates[idx:]
            # If remaining is less than that, we can't reach it? No, we can pick fewer factors.
            # Actually, if we pick k factors, their product is at least prod of k smallest available.
            # But we don't know k.
            
            # Better pruning:
            # If remaining is not divisible by any candidate from idx to end, return False.
            # This check is expensive if done naively.
            
            # Let's just iterate.
            # We can either skip candidates[idx] or take it.
            
            # Option 1: Skip candidates[idx]
            if dfs(idx + 1, remaining, count):
                return True
                
            # Option 2: Take candidates[idx]
            c = candidates[idx]
            if remaining % c == 0:
                if dfs(idx + 1, remaining // c, count + 1):
                    return True
            
            return False
        
        # We need to start with idx=0, remaining=target, count=0.
        # But we must ensure distinctness. By using idx and moving forward, we ensure distinctness.
        # However, this explores subsets in a specific order.
        # Is it possible that we need to skip a small factor to use a larger one? Yes.
        # The DFS handles this by exploring "skip" and "take".
        
        # One optimization: If remaining > product of all remaining candidates, return False.
        # Precompute suffix products?
        # suffix_prod[i] = product of candidates[i:]
        # If remaining > suffix_prod[idx], return False.
        
        n = len(candidates)
        suffix_prod = [1] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_prod[i] = suffix_prod[i + 1] * candidates[i]
            
        # Re-define dfs with suffix_prod access
        # Since lru_cache doesn't take lists, we close over it.
        
        def dfs2(idx, remaining, count):
            if remaining == 1:
                return count > 0
            if count >= 12:
                return False
            if idx >= n:
                return False
            
            # Pruning: Max possible product from idx onwards is suffix_prod[idx]
            if remaining > suffix_prod[idx]:
                return False
            
            # Pruning: Min possible product from idx onwards with (12-count) factors?
            # If we MUST pick more factors to reach 12? No, we can stop early if remaining==1.
            # But if remaining > 1, we MUST pick at least 1 factor.
            # If remaining < candidates[idx], we can't pick any factor >= candidates[idx] that divides remaining?
            # Actually, if remaining < candidates[idx], we can't pick candidates[idx] or any larger.
            # So if remaining > 1 and remaining < candidates[idx], return False.
            if remaining > 1 and remaining < candidates[idx]:
                return False

            # Skip
            if dfs2(idx + 1, remaining, count):
                return True
                
            # Take
            c = candidates[idx]
            if remaining % c == 0:
                if dfs2(idx + 1, remaining // c, count + 1):
                    return True
            
            return False

        return dfs2(0, target, 0)

    # Binary Search
    lo = 0
    hi = len(valid_divs) - 1
    best_max = valid_divs[-1] # Fallback to largest divisor
    
    # We want the smallest valid max.
    # If can_form(valid_divs[mid]) is True, then any larger value is also True (monotonicity).
    # So we search for the first True.
    
    while lo <= hi:
        mid = (lo + hi) // 2
        if can_form(valid_divs[mid]):
            best_max = valid_divs[mid]
            hi = mid - 1
        else:
            lo = mid + 1
            
    # Now we have best_max. We need to return the actual factors.
    # We can re-run the DFS to find the specific set, or modify can_form to return the set.
    # Let's modify the logic to find the set for best_max.
    
    # Re-implement DFS to return the list of factors
    candidates = [d for d in valid_divs if d <= best_max]
    candidates.sort()
    n = len(candidates)
    suffix_prod = [1] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_prod[i] = suffix_prod[i + 1] * candidates[i]

    def find_set(idx, remaining, count, current_set):
        if remaining == 1:
            if count > 0:
                return list(current_set)
            return None
        if count >= 12:
            return None
        if idx >= n:
            return None
        if remaining > suffix_prod[idx]:
            return None
        if remaining > 1 and remaining < candidates[idx]:
            return None

        # Skip
        res = find_set(idx + 1, remaining, count, current_set)
        if res:
            return res

        # Take
        c = candidates[idx]
        if remaining % c == 0:
            current_set.append(c)
            res = find_set(idx + 1, remaining // c, count + 1, current_set)
            if res:
                return res
            current_set.pop()
            
        return None

    result = find_set(0, target, 0, [])
    if result is None:
        # Should not happen if binary search worked correctly
        return [target]
        
    return result

# EVOLVE-BLOCK-END
