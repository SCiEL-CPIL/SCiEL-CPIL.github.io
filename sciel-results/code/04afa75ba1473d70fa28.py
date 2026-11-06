"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import time
    
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    # Known optimal or near-optimal solutions
    # 9 terms is the minimum for min denom 5.
    # Let's try to find a 9-term solution.
    
    # Strategy: Fix the first 7 denominators to 5..11, and solve for the last 2.
    # Or fix 5..10 and solve for 3?
    # 1/5+1/6+1/7+1/8+1/9+1/10 = 0.8428
    # R = 0.1572
    # 1/d + 1/e + 1/f = 0.1572
    # d, e, f > 10.
    
    # Let's use a generic search for N terms, starting from N=9.
    
    def find_solution(n_terms):
        # Use backtracking with pruning
        # State: current_sum (Fraction), min_den, current_list
        target = Fraction(1, 1)
        
        def backtrack(remaining, min_den, current):
            if time.time() - start_time > deadline:
                return None
            if not current:
                return None
            if len(current) == 0:
                if remaining == 0:
                    return []
                return None
            
            if len(current) == n_terms - 1:
                if remaining > 0:
                    # 1/d = remaining
                    # d = 1/remaining
                    d_frac = 1 / remaining
                    if d_frac.denominator == 1:
                        d = d_frac.numerator
                        if d >= min_den and d not in current:
                            return current + [d]
                return None
            
            # Pruning:
            # Max possible sum with remaining terms
            # If remaining > len(current) / min_den, impossible
            if remaining > len(current) / min_den:
                return None
            
            # Min possible sum with remaining terms
            # If remaining < sum(1/i for i in range(max_den, max_den+len(current))) ... hard to bound tightly
            # Simple bound: remaining < len(current) / (10**7) -> always true for small remaining
            
            # Try denominators
            # Heuristic: start from min_den
            # To keep search space small, limit the range.
            # If remaining is large, d must be small.
            # d >= 1/remaining
            # d <= 1 / (remaining / len(current))
            
            max_d = 10000000
            # Estimate max d
            # remaining <= len(current) / d_min
            # d_min >= 1/remaining
            
            # We can iterate d from min_den to some limit
            # Limit: 1/d < remaining / len(current) is not helpful for upper bound of d.
            # Upper bound for d:
            # remaining - 1/d >= 0 => d >= 1/remaining
            # Also, we need to fill the rest with len(current)-1 terms.
            # Max sum of rest is (len(current)-1)/d_next.
            # So remaining - 1/d <= (len(current)-1)/d_next
            # This is complex.
            
            # Simple approach: iterate d from min_den upwards.
            # Stop if 1/d > remaining.
            
            for d in range(min_den, max_d + 1):
                if time.time() - start_time > deadline:
                    return None
                term = Fraction(1, d)
                if term > remaining:
                    break
                if d in current:
                    continue
                new_remaining = remaining - term
                if new_remaining < 0:
                    continue
                res = backtrack(new_remaining, d + 1, current + [d])
                if res:
                    return res
            return None
        
        return backtrack(target, 5, [])

    # Try to find a 9-term solution
    for n in range(9, 15):
        sol = find_solution(n)
        if sol:
            return sol
            
    # Fallback to the original DP approach if search fails
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
