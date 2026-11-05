"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit
    
    # Target: 5/31
    target = Fraction(5, 31)
    
    # We want to find distinct denominators d1 < d2 < ... < dk
    # such that sum(1/di) = 5/31, minimizing max(di) = dk.
    #
    # Strategy:
    # 1. The largest denominator D must be at least ceil(31/5) = 7 (since 1/D <= 5/31).
    #    Actually, since we need sum = 5/31 > 1, we need multiple terms.
    #    Let's think about the minimum possible largest denominator.
    #    If we use denominators 1..M, the max sum is H_M. We need H_M >= 5/31.
    #    But we need exactly 5/31 with distinct denominators.
    #    
    #    Key insight: The greedy algorithm gives an upper bound.
    #    Let's first compute the greedy solution to get an initial upper bound on max denominator.
    
    # Greedy solution for upper bound
    def greedy():
        r = target
        result = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d in result:
                d += 1
            result.append(d)
            r -= Fraction(1, d)
        return result
    
    greedy_sol = greedy()
    best_max = max(greedy_sol)
    best_sol = greedy_sol
    
    # Now search for solutions with max denominator < best_max
    # We'll do a DFS with the largest denominator fixed, trying smaller values.
    
    # For a fixed largest denominator D, we need to find distinct denominators
    # d1 < d2 < ... < d_{k-1} < D such that sum(1/di) + 1/D = 5/31
    # i.e., sum(1/di) = 5/31 - 1/D
    
    # We can use a recursive search. To make it efficient:
    # - Try D from 7 upward (since 1/7 < 5/31, but we need sum >= 5/31)
    #   Actually D must be such that 5/31 - 1/D > 0, so D > 31/5 = 6.2, so D >= 7.
    # - For each D, the remaining sum is S = 5/31 - 1/D.
    # - We need distinct denominators < D summing to S.
    
    # To find if a solution exists with max denominator D, we can use a backtracking search.
    # But this could be slow. Let's use a smarter approach.
    
    # Observation: For small D, the number of possible denominators is small, so search is fast.
    # We'll iterate D from 7 to best_max - 1, and for each, try to find a solution.
    
    def can_complete(remaining, max_denom, used, depth=0):
        """Try to complete the sum 'remaining' using distinct denominators < max_denom.
        Returns a list of denominators if found, else None."""
        if remaining == 0:
            return []
        if remaining < 0:
            return None
        if max_denom <= 1:
            return None
        
        # Lower bound on next denominator: 1/d <= remaining, so d >= ceil(1/remaining)
        # But also d must be > any previously chosen denominator to maintain distinctness and ordering.
        # We enforce d > last_denom to avoid duplicates and ensure distinctness.
        
        # The smallest possible next denominator is max(2, ...) but we track via used set.
        # For efficiency, we know d must satisfy 1/d <= remaining, so d >= ceil(1/remaining) if remaining > 0.
        # Also d < max_denom.
        
        if remaining > 0:
            min_d = max(2, (1 + remaining.numerator - 1) // remaining.numerator)  # ceil(1/remaining)
            # Actually ceil(1/remaining) = ceil(remaining.denominator / remaining.numerator)
            min_d = max(2, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        else:
            return []
        
        max_d = max_denom - 1
        
        if min_d > max_d:
            return None
        
        # Try denominators from min_d to max_d
        # To find a solution quickly, try smaller denominators first (they contribute more)
        for d in range(min_d, max_d + 1):
            if d in used:
                continue
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                continue
            used.add(d)
            result = can_complete(new_remaining, d + 1, used, depth + 1)
            used.remove(d)
            if result is not None:
                return [d] + result
        return None
    
    # This pure backtracking might be too slow. Let's use a better heuristic:
    # For each candidate D (largest denominator), compute S = 5/31 - 1/D.
    # Then try to express S as sum of distinct unit fractions with denominators < D.
    # 
    # A faster approach: use the fact that for the remaining sum, we can try
    # a greedy-like approach but with backtracking only when needed.
    
    # Let's try a different strategy: iterate D from 7 upward, and for each D,
    # try to find a solution using a limited backtracking search.
    
    # But first, let's check if there's a solution with very small max denominator.
    # The absolute minimum: we need sum = 5/31 ≈ 0.1613.
    # With denominators starting from 7: 1/7 ≈ 0.1429, remaining ≈ 0.0184.
    # 1/54 ≈ 0.0185, so {7, 54} gives 1/7 + 1/54 = (54+7)/(7*54) = 61/378 ≈ 0.1614. Close!
    # 61/378 vs 5/31: 61*31 = 1891, 5*378 = 1890. So 61/378 = 1891/11718, 5/31 = 1890/11718.
    # Very close but not exact.
    
    # Let's systematically search. We'll try to find the minimum D such that
    # there exist distinct d1 < d2 < ... < d_{k-1} < D with sum(1/di) = 5/31 - 1/D.
    
    # For efficiency, we'll limit the search depth and use pruning.
    
    found_solution = None
    
    # Try D from 7 to best_max - 1
    for D in range(7, best_max):
        if time.time() > deadline:
            break
        
        S = target - Fraction(1, D)
        if S <= 0:
            continue
        
        # Try to find distinct denominators < D summing to S
        # Use backtracking with pruning
        used = set()
        
        def dfs(rem, max_d):
            if time.time() > deadline:
                return None
            if rem == 0:
                return []
            if rem < 0:
                return None
            if max_d <= 1:
                return None
            
            # Next denominator d must satisfy: 1/d <= rem, so d >= ceil(1/rem)
            if rem > 0:
                min_d = max(2, (rem.denominator + rem.numerator - 1) // rem.numerator)
            else:
                return []
            
            if min_d > max_d:
                return None
            
            # Try d from min_d to max_d
            # For speed, try a limited number of candidates
            # If rem is small, we need large d
            for d in range(min_d, max_d + 1):
                if d in used:
                    continue
                new_rem = rem - Fraction(1, d)
                if new_rem < 0:
                    continue
                used.add(d)
                res = dfs(new_rem, d + 1)
                used.remove(d)
                if res is not None:
                    return [d] + res
            return None
        
        result = dfs(S, D)
        if result is not None:
            sol = result + [D]
            # Verify
            total = sum(Fraction(1, x) for x in sol)
            if total == target:
                found_solution = sol
                best_max = D
                best_sol = sol
                break  # We found the minimum D
    
    if found_solution is None:
        return best_sol
    
    return best_sol
# EVOLVE-BLOCK-END
