def construct():
    import time
    from math import factorial
    
    target = factorial(12)  # 479001600
    deadline = time.time() + 85  # 85s safety margin (limit is 90s)
    
    # Prime factorization of 12!: 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    # Generate all divisors of target
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divs = []
        for d in divisors:
            val = 1
            for _ in range(e + 1):
                new_divs.append(d * val)
                val *= p
        divisors = new_divs
    
    # Filter divisors > 12 and < target (proper divisors for intermediate steps, 
    # but we also need to consider target itself if it's the only factor, though 
    # distinctness and >12 constraint applies. If k=1, factor is target. 
    # But we want to minimize max, so k=1 is bad unless no other option.
    # Heuristic will likely find k>1.)
    
    # We need divisors > 12.
    valid_divisors = [d for d in divisors if d > 12]
    valid_divisors.sort()
    
    best = [target]
    best_max = target
    
    # Heuristic: Greedy construction
    # Try to find a decomposition with small max factor.
    # Strategy: Pick the largest divisor of 'remaining' that is > min_factor and < best_max.
    # If no such divisor exists, we must use 'remaining' as the last factor (if valid).
    
    def heuristic_search(remaining, min_factor, chosen):
        nonlocal best, best_max
        if remaining == 1:
            if chosen:
                max_f = max(chosen)
                if max_f < best_max:
                    best = chosen[:]
                    best_max = max_f
            return
        
        if len(chosen) >= 12:
            return
            
        # If remaining is a valid factor and distinct, we can stop here
        if remaining > min_factor and remaining not in chosen:
            max_f = max(max(chosen) if chosen else 0, remaining)
            if max_f < best_max:
                best = chosen + [remaining]
                best_max = max_f
            return # No need to split further if we found a valid complete solution? 
                   # Actually, splitting might yield a smaller max? 
                   # No, splitting a number $X$ into $A, B$ with $A,B > 1$ implies $A,B < X$.
                   # So the max of the set will decrease or stay same? 
                   # If we replace $X$ with $A, B$, the new max is $\max(\max(\text{prev}), A, B)$.
                   # Since $A, B < X$, if $X$ was the max, the new max is smaller.
                   # So we should ALWAYS try to split if possible to reduce max?
                   # Yes, unless we are constrained by distinctness or count.
                   # But if we can't split into distinct factors > min_factor, we must keep it.
                   
        # Try to split 'remaining'
        # Iterate divisors of 'remaining' that are > min_factor and < best_max
        # We want to pick a divisor d. The other part is remaining//d.
        # To minimize the max, we should pick d close to sqrt(remaining) if possible?
        # Or just pick any valid d that leads to a better solution.
        # Greedy: Pick the largest valid d < best_max? 
        # Or the d closest to sqrt(remaining)?
        # Let's try to find ANY decomposition that improves best_max.
        
        # Find all divisors of remaining that are > min_factor and < best_max
        # Note: remaining//d must also be > min_factor (if we stop) or splittable.
        # For the heuristic, let's just try to find a split that reduces the max.
        
        # We need to be careful with time.
        if time.time() > deadline:
            return
            
        # Get divisors of remaining
        # Since remaining is a divisor of target, its divisors are in 'valid_divisors' 
        # (plus 1 and itself).
        # We can filter valid_divisors for those dividing remaining.
        # This is O(N) where N is number of divisors of target (~ 11*6*3*2*2 = 792).
        # This is fast.
        
        candidates = []
        for d in valid_divisors:
            if d > min_factor and d < best_max and remaining % d == 0:
                other = remaining // d
                # 'other' must be > min_factor to be a valid single factor, 
                # or we recurse.
                # If we recurse, we don't need other > min_factor immediately, 
                # but the factors of other must be.
                # However, if we treat 'other' as a single factor in this step, 
                # it must be > min_factor and distinct.
                # Let's just recurse on 'other' with min_factor = d+1 (to ensure distinctness? 
                # No, distinctness is global. We just need to ensure no duplicates in 'chosen' + [d] + factors_of_other.
                # The DFS below handles distinctness via min_factor and set check? 
                # The current DFS uses min_factor to enforce sorted order, which ensures distinctness.
                # So if we recurse, we pass min_factor = d+1.
                
                # Prune: if other is 1, we found a solution.
                # If other > best_max, and we can't split it further to reduce max below best_max?
                # Actually, if other > best_max, and we use d, the max will be at least other (if we stop) 
                # or factors of other. If factors of other are all < best_max, it's fine.
                # So we can recurse even if other > best_max, as long as we can split other.
                
                candidates.append(d)
        
        # Sort candidates to try "balanced" splits first? 
        # Smaller d means larger other. Larger d means smaller other.
        # To minimize max, we want both d and factors of other to be small.
        # Picking d close to sqrt(remaining) is a good heuristic for minimizing max(d, other).
        
        # Let's just iterate through candidates.
        # To make it fast, we can limit the number of candidates tried in heuristic.
        
        for d in candidates:
            if time.time() > deadline:
                break
            chosen.append(d)
            heuristic_search(remaining // d, d + 1, chosen)
            chosen.pop()
            
            # If we found a solution with max < best_max, we can stop? 
            # No, we want the MINIMUM. But heuristic is just for initialization.
            # We can stop if best_max becomes very small, but let's let it run a bit.
            # Actually, if we find a solution, we update best_max. 
            # Subsequent branches in this heuristic call will use the tighter best_max.
            
    heuristic_search(target, 13, [])
    
    # Now run the exact DFS with the improved best_max
    divisors_sorted = valid_divisors
    
    def dfs(remaining, chosen, min_factor, start_idx):
        nonlocal best, best_max
        if time.time() > deadline:
            return
        
        if remaining == 1:
            if chosen:
                max_f = max(chosen)
                if max_f < best_max:
                    best = chosen[:]
                    best_max = max_f
            return
        
        if len(chosen) >= 12:
            return
            
        # Pruning: 
        # 1. If we can form 'remaining' with a single factor d > min_factor and d not in chosen:
        #    Then max is max(max(chosen), d). If this < best_max, update.
        #    But we can also split.
        
        # 2. Lower bound on max factor:
        #    If we have k slots left (12 - len(chosen)), the product is 'remaining'.
        #    The factors must be distinct and > min_factor.
        #    The smallest possible max factor is at least the k-th root of remaining?
        #    Roughly. A simple bound: if remaining > best_max, we MUST split.
        #    If remaining <= best_max, we could stop, but maybe splitting yields smaller max?
        #    Yes, if remaining can be split into factors all < best_max.
        
        # Iterate through divisors
        for i in range(start_idx, len(divisors_sorted)):
            d = divisors_sorted[i]
            if d < min_factor:
                continue
            if d > remaining:
                break
            if remaining % d != 0:
                continue
            
            # Prune: if d >= best_max, we can't improve (since max will be at least d)
            # UNLESS d == remaining and we are stopping? 
            # If d == remaining, max is max(max(chosen), d).
            # If d >= best_max, then max >= best_max. No improvement.
            # So we can break if d >= best_max.
            if d >= best_max:
                break
                
            chosen.append(d)
            dfs(remaining // d, chosen, d + 1, i + 1)
            chosen.pop()
            
    dfs(target, [], 13, 0)
    
    return best
