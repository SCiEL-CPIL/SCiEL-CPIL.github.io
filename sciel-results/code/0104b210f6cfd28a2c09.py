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
    
    # Filter divisors > 12
    valid_divs = sorted([d for d in divisors if d > 12])
    
    # We want to find a subset of distinct elements from valid_divs
    # whose product is target, minimizing the max element.
    # Use DFS with pruning.
    
    best_max = target  # single factor case
    best_set = [target]
    
    n = len(valid_divs)
    
    # Sort in descending order to find large factors first? 
    # Actually, we want to minimize the max, so let's think differently.
    # 
    # Approach: For a given maximum M, check if we can find a valid set with all elements <= M.
    # Binary search on M? But the set of possible M values is the divisors themselves.
    #
    # Alternative: DFS that tries to build a set, pruning if current product * remaining_min > target
    # or if we exceed 12 factors.
    
    # Let's use a different approach: 
    # Sort valid_divs in ascending order.
    # Use DFS: at each step, choose the next factor from remaining divisors.
    # Prune: if current_product > target, backtrack.
    # Prune: if len(chosen) >= 12, check if current_product == target.
    # Prune: if current_product * product_of_smallest_remaining < target and we've used all... 
    # This is tricky.
    
    # Better: Since we want to minimize max, let's iterate over possible max values (divisors > 12)
    # in ascending order, and for each max M, check if there exists a valid set with all elements <= M
    # and max exactly M (or <= M).
    
    # For a fixed M, filter valid_divs to those <= M, and check if any subset (size 1-12) 
    # of distinct elements has product = target.
    
    # This is still NP-hard in general, but with only ~700 divisors and product constraint, 
    # we can use DFS with strong pruning.
    
    # Let's try: for each candidate max M (in ascending order), do a DFS to see if we can
    # form target using distinct divisors from {d in valid_divs : d <= M}.
    # The first M for which we find a solution gives us the answer.
    
    # To make DFS efficient, sort candidates in descending order within the DFS.
    
    def can_form_target(candidates, target, max_factors=12):
        """DFS to check if we can form target using distinct elements from candidates."""
        # candidates is a sorted list (we'll sort in descending order for DFS)
        n = len(candidates)
        if n == 0:
            return target == 1
        
        # Quick check: product of all candidates must be >= target (if we use all)
        # But we don't have to use all.
        
        # Pruning: if product of all candidates < target, impossible
        prod_all = 1
        for c in candidates:
            prod_all *= c
            if prod_all > target:
                break
        if prod_all < target:
            return False
        
        # If target is in candidates, we could use just that one (if target > 12 and in list)
        # But we need distinct, and we're building a set.
        
        # DFS: include or exclude each candidate
        # Sort candidates in descending order to find large factors first (helps with pruning)
        candidates = sorted(candidates, reverse=True)
        
        def dfs(idx, current_prod, count):
            if current_prod > target:
                return False
            if count > max_factors:
                return False
            if current_prod == target:
                return count >= 1
            if idx >= n:
                return False
            
            # Pruning: even if we multiply by all remaining candidates, can we reach target?
            # This is expensive to compute each time. Alternative pruning:
            # If current_prod * (product of all remaining from idx to end) < target, prune.
            # But computing that each time is O(n). Let's precompute suffix products? 
            # But they can overflow. Use logarithms or just check a few.
            
            # Simpler pruning: if current_prod > target, already handled.
            # If we have too few factors left to reach target... hard to say.
            
            # Try including candidates[idx]
            if current_prod * candidates[idx] <= target:
                if dfs(idx + 1, current_prod * candidates[idx], count + 1):
                    return True
            
            # Try excluding candidates[idx]
            if dfs(idx + 1, current_prod, count):
                return True
            
            return False
        
        return dfs(0, 1, 0)
    
    # Iterate over possible max values in ascending order
    # But this could be slow. Let's limit the search.
    
    # Actually, let's just do a single DFS that tracks the best solution found,
    # with pruning based on current max.
    
    # Let me try a different, more practical approach:
    # Use a greedy/heuristic search.
    
    # Start with target, try to divide it into factors > 12.
    # Use a priority queue or iterative deepening.
    
    # Given time constraints, let me try a focused search:
    # Try factor counts from 2 to 12, and for each, try to find a good decomposition.
    
    # For a given number of factors k, we want to find k distinct divisors > 12
    # whose product is target, minimizing the max.
    
    # Let's use a randomized/local search approach combined with some deterministic steps.
    
    # Actually, let me just do a bounded DFS with good pruning.
    
    # Sort valid_divs in ascending order
    valid_divs_sorted = sorted(valid_divs)
    
    # For efficiency, let's limit the DFS depth and use memoization on (idx, current_prod)
    # but current_prod can be large. Instead, let's use the fact that we're dividing target.
    
    # Reformulate: we want to find distinct d1, d2, ..., dk from valid_divs such that
    # d1 * d2 * ... * dk = target.
    
    # Equivalently, let target = d1 * d2 * ... * dk.
    # Start with remaining = target. At each step, pick a divisor d of remaining, d > 12,
    # d not used before, and recurse with remaining // d.
    
    # This is cleaner!
    
    used = set()
    
    def dfs(remaining, min_factor, count):
        """Find distinct factors > 12, each >= min_factor (to ensure distinctness and ordering),
        whose product is remaining. Returns the list of factors or None."""
        if remaining == 1:
            return []
        if count >= 12:
            return None
        
        # Find divisors of remaining that are > 12, >= min_factor, and not in used
        # To minimize max, we want smaller factors first? No, we want to explore
        # all options but prune if current factor is too large.
        
        # Get divisors of remaining
        divs = []
        for i in range(1, isqrt(remaining) + 1):
            if remaining % i == 0:
                if i >= min_factor and i > 12:
                    divs.append(i)
                other = remaining // i
                if other != i and other >= min_factor and other > 12:
                    divs.append(other)
        
        # Sort divisors in ascending order to try smaller factors first
        # (this helps find solutions with smaller max factors)
        divs = sorted(set(divs))
        
        for d in divs:
            if d in used:
                continue
            used.add(d)
            result = dfs(remaining // d, d + 1, count + 1)
            if result is not None:
                return [d] + result
            used.remove(d)
        
        return None
    
    # Try to find a solution
    # To minimize the max factor, we should try to use as many factors as possible
    # with small values. Let's try different strategies.
    
    # Strategy: DFS with ascending order of factors, which naturally tends to
    # find solutions with smaller maximum factors.
    
    result = dfs(target, 13, 0)
    
    if result is not None:
        # We found a solution. But is it optimal?
        # The DFS with ascending order doesn't guarantee optimality.
        # Let's verify and possibly improve.
        # For now, return this solution.
        return result
    else:
        # Fallback: single factor
        return [target]
    
# EVOLVE-BLOCK-END
