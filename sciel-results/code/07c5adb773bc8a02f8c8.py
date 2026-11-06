from fractions import Fraction

def construct():
    # We proved N < 8 is impossible for denominators >= 5.
    # We search for N = 8.
    target = Fraction(1, 1)
    result = search_8(target, 5, [])
    if result:
        return result
    # Fallback if 8 fails (should not happen), try 9, etc.
    for n in range(9, 20):
        result = search_n(target, 5, n, [])
        if result:
            return result
    raise RuntimeError("no representation found")

def search_8(remaining, min_denom, chosen):
    # Optimized for exactly 8 terms
    return search_n(remaining, min_denom, 8, chosen)

def search_n(remaining, min_denom, terms_left, chosen):
    if terms_left == 0:
        return chosen if remaining == 0 else None
    
    if remaining <= 0:
        return None
        
    # Pruning 1: Max possible sum with terms_left terms starting from min_denom
    # The largest sum is achieved by taking the smallest available denominators:
    # min_denom, min_denom+1, ..., min_denom + terms_left - 1
    max_sum = sum(Fraction(1, i) for i in range(min_denom, min_denom + terms_left))
    if max_sum < remaining:
        return None
        
    # Pruning 2: Min possible sum is 0, but we can use a lower bound on denominators?
    # If we pick a denominator d, the next denominators must be > d.
    # The sum of the remaining terms must be achievable.
    
    # Determine the range for the next denominator d
    # 1/d <= remaining  => d >= 1/remaining
    # Also d >= min_denom
    
    # Upper bound for d:
    # Even if we take the smallest possible remaining terms (d+1, d+2, ...),
    # the sum must be >= remaining - 1/d.
    # Actually, we just iterate d. To make it fast, we can limit the range of d.
    # d cannot be too large, otherwise 1/d is too small and we can't reach the target
    # with the remaining terms? No, if 1/d is small, remaining - 1/d is large,
    # and we have fewer terms.
    # The constraint is: remaining - 1/d must be <= max_sum(terms_left-1, d+1).
    # max_sum(k, start) = sum(1/i for i in range(start, start+k))
    
    # So: remaining - 1/d <= sum(1/i for i in range(d+1, d+1+terms_left-1))
    # 1/d >= remaining - max_sum(terms_left-1, d+1)
    
    # Let's just iterate d from min_denom up to a reasonable limit.
    # Since N is small (8), the search tree is small.
    
    # Start d
    if remaining < 1:
        min_d = max(min_denom, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
    else:
        min_d = min_denom
        
    # End d
    # We can estimate an upper bound.
    # If d is very large, 1/d is small.
    # Let's just set a generous upper bound, e.g., min_denom + 1000.
    # Or calculate it dynamically.
    
    for d in range(min_d, min_denom + 1000):
        if d < min_denom:
            continue
        if 1 / d > remaining:
            continue
            
        new_remaining = remaining - Fraction(1, d)
        if new_remaining < 0:
            break
            
        # Check if new_remaining is achievable with terms_left-1 terms > d
        # Max sum with terms_left-1 terms starting from d+1
        if terms_left > 1:
             max_next_sum = sum(Fraction(1, i) for i in range(d+1, d+1+terms_left-1))
             if max_next_sum < new_remaining:
                 # If even the smallest denominators can't reach the remaining sum,
                 # then no larger denominators can. So we can break?
                 # No, larger denominators give SMALLER sums.
                 # So if max_sum < target, it's impossible.
                 # Since max_sum decreases as d increases (start index increases),
                 # we can break.
                 break
                 
        result = search_n(new_remaining, d + 1, terms_left - 1, chosen + [d])
        if result is not None:
            return result
            
    return None
