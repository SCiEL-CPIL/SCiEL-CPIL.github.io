"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd

def _try_complete_greedy(rem, max_terms=256, deadline=None):
    """Complete the remainder with greedy Egyptian fractions, enforcing distinctness."""
    if rem == 0:
        return []
    values = []
    used = set()
    r = rem
    while r != 0:
        if deadline is not None and time.time() > deadline:
            return None
        if len(values) >= max_terms:
            return None
        # Greedy step
        d = (r.denominator + r.numerator - 1) // r.numerator
        # Ensure distinct
        while d in used:
            d += 1
        if d > 10_000_000:
            return None
        values.append(d)
        used.add(d)
        r -= Fraction(1, d)
    return values

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin for 90s limit
    
    target = Fraction(5, 31)
    best_max = 10_000_000  # Initialize with a large value
    best_solution = None
    
    def find_solution_with_max_limit(L, max_terms, deadline):
        """
        Find a solution with all denominators <= L.
        Returns list of denominators or None.
        """
        # Recursive helper
        def rec(rem, min_d, terms_left, current):
            if rem == 0:
                return list(current) if current else None
            if terms_left == 0:
                return None
            if rem < 0:
                return None
            
            # If terms_left == 1
            if terms_left == 1:
                if rem.denominator % rem.numerator == 0:
                    d = rem.denominator // rem.numerator
                    if min_d <= d <= L:
                        current.append(d)
                        res = list(current)
                        current.pop()
                        return res
                return None
            
            # Determine range for next denominator d
            # d must satisfy: 1/d < rem (if terms_left > 1)
            # So d > 1/rem => d >= floor(1/rem) + 1
            # Also d >= min_d
            
            # Upper bound: rem - 1/d must be formable by terms_left-1 fractions with denominators > d
            # Max sum of terms_left-1 fractions with denominators > d is (terms_left-1)/(d+1)
            # So rem - 1/d <= (terms_left-1)/(d+1)
            
            # Find minimum d
            d_min = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
            
            d = d_min
            while d <= L:
                if time.time() > deadline:
                    return None
                
                rem2 = rem - Fraction(1, d)
                
                if rem2 < 0:
                    # 1/d > rem, need larger d
                    d += 1
                    continue
                
                if rem2 == 0:
                    # Can't use 0 more terms
                    d += 1
                    continue
                
                # Check if rem2 can be formed by terms_left-1 terms with denominators in [d+1, L]
                # Max possible sum is (terms_left-1)/(d+1)
                max_possible = Fraction(terms_left - 1, d + 1)
                if rem2 > max_possible:
                    # For larger d, rem2 increases and max_possible decreases, so break
                    break
                
                # Min possible sum is 0 (not helpful for pruning)
                # But we can check if rem2 is too small to be formed by terms_left-1 distinct denominators >= d+1
                # The smallest sum with terms_left-1 distinct denominators >= d+1 is approximately
                # 1/(d+1) + 1/(d+2) + ... but this is complex. Skip for now.
                
                current.append(d)
                res = rec(rem2, d + 1, terms_left - 1, current)
                current.pop()
                if res is not None:
                    return res
                d += 1
            return None

        return rec(target, 1, max_terms, [])

    # Strategy: Find the minimum L such that a solution exists
    # First, find any solution to get an upper bound
    # Then binary search for the minimum L
    
    # Step 1: Find an initial solution with a reasonable L
    initial_L = 1000
    sol = None
    for terms in range(2, 8):
        if time.time() > deadline:
            break
        sol = find_solution_with_max_limit(initial_L, terms, deadline)
        if sol:
            # Validate
            s = sum(Fraction(1, d) for d in sol)
            if s == target and len(sol) == len(set(sol)) and all(1 <= d <= 10_000_000 for d in sol):
                best_max = max(sol)
                best_solution = sol
                break
    
    if best_solution is None:
        # Fallback: try larger L
        for L in [2000, 5000, 10000]:
            if time.time() > deadline:
                break
            for terms in range(2, 8):
                if time.time() > deadline:
                    break
                sol = find_solution_with_max_limit(L, terms, deadline)
                if sol:
                    s = sum(Fraction(1, d) for d in sol)
                    if s == target and len(sol) == len(set(sol)) and all(1 <= d <= 10_000_000 for d in sol):
                        best_max = max(sol)
                        best_solution = sol
                        break
            if best_solution:
                break
    
    if best_solution is None:
        # Greedy fallback
        r = target
        values = []
        used = set()
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            while d in used:
                d += 1
            values.append(d)
            used.add(d)
            r -= Fraction(1, d)
        best_solution = values
        best_max = max(values)
        return best_solution
    
    # Step 2: Binary search for the minimum L
    # We know a solution exists with max <= best_max
    # Search in [1, best_max]
    
    low = 1
    high = best_max
    
    # First, try small values directly to avoid missing the optimum
    # Try L from 10 to min(best_max, 100) in steps of 1
    for L in range(10, min(best_max + 1, 101)):
        if time.time() > deadline:
            break
        for terms in range(2, 8):
            if time.time() > deadline:
                break
            sol = find_solution_with_max_limit(L, terms, deadline)
            if sol:
                s = sum(Fraction(1, d) for d in sol)
                if s == target and len(sol) == len(set(sol)) and all(1 <= d <= 10_000_000 for d in sol):
                    if max(sol) < best_max:
                        best_max = max(sol)
                        best_solution = sol
                        high = best_max
                        # Don't break, keep searching for smaller L
                break
    
    # Now binary search in [low, high] if high > 100
    if high > 100 and time.time() < deadline:
        while low < high and time.time() < deadline:
            mid = (low + high) // 2
            found = False
            for terms in range(2, 8):
                if time.time() > deadline:
                    break
                sol = find_solution_with_max_limit(mid, terms, deadline)
                if sol:
                    s = sum(Fraction(1, d) for d in sol)
                    if s == target and len(sol) == len(set(sol)) and all(1 <= d <= 10_000_000 for d in sol):
                        if max(sol) < best_max:
                            best_max = max(sol)
                            best_solution = sol
                        high = mid
                        found = True
                        break
            if not found:
                low = mid + 1
    
    # Step 3: Local search around the best found
    # Try L from best_max - 20 to best_max
    for L in range(max(1, best_max - 20), best_max + 1):
        if time.time() > deadline:
            break
        for terms in range(2, 8):
            if time.time() > deadline:
                break
            sol = find_solution_with_max_limit(L, terms, deadline)
            if sol:
                s = sum(Fraction(1, d) for d in sol)
                if s == target and len(sol) == len(set(sol)) and all(1 <= d <= 10_000_000 for d in sol):
                    if max(sol) < best_max:
                        best_max = max(sol)
                        best_solution = sol
                break
    
    return best_solution

# EVOLVE-BLOCK-END
