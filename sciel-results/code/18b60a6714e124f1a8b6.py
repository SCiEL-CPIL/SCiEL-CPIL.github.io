def construct():
    import time
    from fractions import Fraction
    from math import isqrt

    deadline = time.time() + 80  # 80s safety margin (90s hard limit)
    target = Fraction(5, 31)

    # Known good solution from seed (greedy)
    def greedy():
        r = target
        vals = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            vals.append(d)
            r -= Fraction(1, d)
        return vals

    best = greedy()
    best_max = max(best)

    # DFS search for solutions with max denominator < best_max
    # We search for representations as sum of distinct unit fractions
    # with all denominators <= max_den
    
    def dfs_search(max_den, time_limit):
        """Find a set of distinct denominators in [1, max_den] summing to target."""
        nonlocal best, best_max
        start = time.time()
        
        # Pruning: at each step, we need remaining sum = r
        # With remaining denominators from d_min to max_den
        # The maximum possible sum from remaining denominators (d_min, d_min+1, ..., max_den)
        # is bounded, and the minimum is 1/max_den (if we could use just one more)
        
        def dfs(remaining_sum, start_den, chosen, depth):
            nonlocal best, best_max
            if time.time() - start > time_limit:
                return None
            
            if remaining_sum == 0:
                if chosen:
                    return list(chosen)
                return None
            
            if remaining_sum < 0:
                return None
            
            # Lower bound on next denominator:
            # 1/d <= remaining_sum => d >= 1/remaining_sum
            min_d = max(start_den, (remaining_sum.denominator + remaining_sum.numerator - 1) // remaining_sum.numerator)
            if min_d > max_den:
                return None
            
            # Upper bound: even if we use all denominators from min_d to max_den,
            # can we reach remaining_sum?
            # Max sum from denominators min_d..max_den is roughly sum(1/i for i in min_d..max_den)
            # This is about ln(max_den/min_d). If remaining_sum > that, impossible.
            # But computing exactly is expensive. Use a looser bound:
            # The number of terms we can still add is at most max_den - min_d + 1
            # Each term is at most 1/min_d, so max remaining sum <= (max_den - min_d + 1) / min_d
            max_terms = max_den - min_d + 1
            if remaining_sum > max_terms / min_d:
                return None
            
            # Also, with just one more term: 1/d = remaining_sum => d = 1/remaining_sum
            # If d is an integer and d >= min_d and d <= max_den, we have a solution
            if remaining_sum.numerator == 1:
                d = remaining_sum.denominator
                if min_d <= d <= max_den:
                    return list(chosen) + [d]
            
            # Try each possible next denominator
            for d in range(min_d, max_den + 1):
                new_sum = remaining_sum - Fraction(1, d)
                if new_sum < 0:
                    break  # Since d is increasing, new_sum will only get more negative
                if new_sum == 0:
                    return list(chosen) + [d]
                
                result = dfs(new_sum, d + 1, chosen + [d], depth + 1)
                if result is not None:
                    return result
                if time.time() - start > time_limit:
                    return None
            return None
        
        return dfs(target, 2, [], 0)

    # Try to find a solution with progressively smaller max denominators
    # Start from best_max - 1 and go down
    search_time_per_try = 5  # seconds per search attempt
    for candidate_max in range(best_max - 1, 1, -1):
        if time.time() > deadline - 10:
            break
        result = dfs_search(candidate_max, search_time_per_try)
        if result is not None and len(result) >= 1 and len(result) <= 256:
            # Verify
            s = sum(Fraction(1, d) for d in result)
            if s == target and all(1 <= d <= 10000000 for d in result) and len(set(result)) == len(result):
                best = result
                best_max = candidate_max
                # Continue trying to go lower
                search_time_per_try = min(search_time_per_try + 1, 15)
            else:
                break
        else:
            # No solution found at this max; we could try slightly larger, 
            # but we're going downward so just continue
            # Actually if no solution at candidate_max, there might still be at lower values?
            # No! If no solution with max <= candidate_max, then no solution with max < candidate_max either
            # Wait, that's not right. We're searching for max denominator exactly candidate_max,
            # meaning all denominators <= candidate_max. If no solution exists with all denoms <= candidate_max,
            # then certainly no solution with all denoms < candidate_max.
            # So we can stop.
            break

    return best
