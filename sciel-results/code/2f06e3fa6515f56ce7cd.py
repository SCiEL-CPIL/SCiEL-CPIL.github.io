import time
from fractions import Fraction

def _find_solution(n_terms, min_den=5, max_den=10000000, deadline=None):
    """
    Find n_terms distinct denominators >= min_den whose reciprocal sum is exactly 1.
    Strategy: pick (n_terms-2) denominators, solve for last 2.
    """
    if deadline is None:
        deadline = time.time() + 50
    
    target = Fraction(1, 1)
    pick_count = n_terms - 2
    
    def try_solve_last2(dens):
        """Try to solve for last 2 given (n_terms-2) denominators."""
        current_sum = sum(Fraction(1, d) for d in dens)
        if current_sum >= target:
            return None
        R = target - current_sum
        
        min_a = dens[-1] + 1
        # a must satisfy: 1/a + 1/b = R, b > a
        # So 1/a > R/2, meaning a < 2/R
        # And 1/a < R, meaning a > 1/R
        low_a = max(min_a, int(1 / R) + 1)
        high_a = min(int(2 / R), max_den)
        
        if high_a < low_a:
            return None
        
        dens_set = set(dens)
        for a in range(low_a, high_a + 1):
            if a in dens_set:
                continue
            # b = a / (R*a - 1)
            num = a
            den = R * a - 1
            if den <= 0:
                continue
            if num % den == 0:
                b = num // den
                if b > a and b <= max_den and b not in dens_set:
                    return dens + [a, b]
        return None
    
    def search(dens, idx, partial_sum):
        if time.time() > deadline:
            return None
        
        if idx == pick_count:
            return try_solve_last2(dens)
        
        start = dens[idx - 1] + 1 if idx > 0 else min_den
        
        # Bound: remaining terms must sum to (target - partial_sum)
        remaining = target - partial_sum
        terms_left = pick_count - idx
        
        # Max possible sum with terms_left terms starting from start
        # 1/start + 1/(start+1) + ... + 1/(start+terms_left-1)
        max_possible = sum(Fraction(1, start + i) for i in range(terms_left))
        if partial_sum + max_possible < target:
            return None
        
        # Min possible sum: use largest denominators, but they're unbounded above
        # So just check if we can still reach target
        if partial_sum >= target:
            return None
        
        # Estimate range for next denominator
        # 1/d should be roughly remaining / terms_left
        # d should be around terms_left / remaining
        approx_d = int(terms_left / remaining) + 1
        if approx_d < start:
            approx_d = start
        
        # Search window around approx_d
        window = 200
        lo = max(start, approx_d - window)
        hi = min(max_den, approx_d + window)
        
        for d in range(lo, hi + 1):
            if time.time() > deadline:
                return None
            
            new_partial = partial_sum + Fraction(1, d)
            if new_partial >= target:
                break
            
            # Check if remaining terms can still reach target
            rem = target - new_partial
            left = terms_left - 1
            if left > 0:
                max_rem = sum(Fraction(1, d + 1 + i) for i in range(left))
                if new_partial + max_rem < target:
                    break  # even with smallest remaining denominators, can't reach 1
            
            result = search(dens + [d], idx + 1, new_partial)
            if result is not None:
                return result
        
        return None
    
    return search([], 0, Fraction(0, 1))

def construct():
    deadline = time.time() + 55
    
    # Try to find minimal solutions starting from 8 terms
    for n in range(8, 20):
        res = _find_solution(n, deadline=deadline)
        if res is not None:
            # Verify
            s = sum(Fraction(1, d) for d in res)
            if s == Fraction(1, 1) and len(res) == len(set(res)) and all(5 <= d <= 10000000 for d in res):
                return res
    
    # Fallback: use a verified solution
    # Construct using splitting method to ensure distinct denominators
    # Start: 1 = 1/5 + 1/5 + 1/5 + 1/5 + 1/5
    # Split 1/5 = 1/10 + 1/10, then 1/10 = 1/20 + 1/20, etc.
    # This gets complex. Use DP fallback.
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
