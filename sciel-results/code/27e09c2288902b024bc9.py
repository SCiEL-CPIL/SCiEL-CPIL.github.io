"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin before 90s kill

    # Known small solutions to try first (fewer terms is better)
    # Each is a list of denominators >= 5, distinct, sum of reciprocals = 1
    # We'll verify them and return the shortest valid one immediately.
    
    # Try to find solutions with few terms via a targeted search.
    # The minimum number of terms for 1 = sum of 1/d_i with d_i >= 5, distinct:
    # - 5 terms: max possible sum with 5 distinct denominators >= 5 is 1/5+1/6+1/7+1/8+1/9 ≈ 0.645 < 1. Impossible.
    # - 6 terms: 1/5+1/6+1/7+1/8+1/9+1/10 ≈ 0.716 < 1. Impossible.
    # - 7 terms: 1/5+1/6+1/7+1/8+1/9+1/10+1/11 ≈ 0.771 < 1. Impossible.
    # - 8 terms: 1/5+1/6+1/7+1/8+1/9+1/10+1/11+1/12 ≈ 0.818 < 1. Impossible.
    # - 9 terms: 1/5+1/6+1/7+1/8+1/9+1/10+1/11+1/12+1/13 ≈ 0.857 < 1. Still < 1!
    # - 10 terms: add 1/14 ≈ 0.889 < 1. Still < 1!
    # - We need more terms. Let's compute: harmonic sum from 5 to n.
    #   H(14) - H(4) = (1+1/2+1/3+...+1/14) - (1+1/2+1/3+1/4)
    #   H(14) ≈ 3.2516, H(4) ≈ 2.0833, diff ≈ 1.168 > 1. So 10 terms can work.
    #   Actually let's check: 1/5+1/6+1/7+1/8+1/9+1/10+1/11+1/12+1/13+1/14
    #   = (1/5+1/6+1/7+1/8+1/9+1/10) + (1/11+1/12+1/13+1/14)
    #   ≈ 0.7163 + 0.3155 ≈ 1.0318 > 1. So 10 terms is possible.
    #   Can we do 9 terms? Max 9-term sum: 1/5+...+1/13 ≈ 0.857 < 1. No.
    #   So minimum is 10 terms.
    
    # Let's find a valid 10-term solution.
    # Start with greedy: pick smallest available denominator >= 5 such that remaining sum is achievable.
    
    def find_solution(num_terms, max_denom=10000000, time_limit=None):
        """Find num_terms distinct denominators >= 5 summing to 1."""
        if time_limit is None:
            time_limit = deadline - time.time() - 2.0
        if time_limit <= 0:
            return None
        
        target = Fraction(1, 1)
        result = []
        
        def dfs(remaining, k, min_denom):
            if time.time() > deadline - 5.0:
                return None
            if k == 0:
                if remaining == 0:
                    return []
                return None
            if remaining <= 0:
                return None
            
            # Lower bound: even with k terms of smallest possible denominators, can we reach 'remaining'?
            # Upper bound: 1/min_denom + 1/(min_denom+1) + ... + 1/(min_denom+k-1) >= remaining?
            # If not, no solution.
            upper_bound = sum(Fraction(1, min_denom + i) for i in range(k))
            if upper_bound < remaining:
                return None
            
            # Also, remaining must be >= 1/(max possible denom) * k ... but that's loose.
            # More useful: remaining must be <= 1/min_denom (if k=1) or generally bounded.
            
            # Try denominators from min_denom upward
            # For efficiency, we can bound the max denominator to try
            # 1/d >= remaining / k  =>  d <= k / remaining
            max_d = int(k / float(remaining)) + 1
            max_d = min(max_d, max_denom)
            
            for d in range(min_denom, max_d + 1):
                if d < 5:
                    continue
                new_remaining = remaining - Fraction(1, d)
                if new_remaining < 0:
                    break
                sub = dfs(new_remaining, k - 1, d + 1)
                if sub is not None:
                    return [d] + sub
            return None
        
        return dfs(target, num_terms, 5)
    
    # Try 10 terms first (proven minimum)
    solution = find_solution(10)
    if solution is not None and len(solution) == 10:
        # Verify
        s = sum(Fraction(1, d) for d in solution)
        if s == 1 and len(set(solution)) == 10 and all(5 <= d <= 10000000 for d in solution):
            return solution
    
    # If 10-term search failed (timeout or no solution found), try known solutions
    # Let's try a constructive approach for 10 terms
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 - excess
    # Sum of 1/5..1/14:
    s = sum(Fraction(1, d) for d in range(5, 15))
    # s = 1/5+1/6+1/7+1/8+1/9+1/10+1/11+1/12+1/13+1/14
    # Let's compute: this is > 1, so we need to replace some terms.
    
    # Alternative: use the identity approach
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 - excess
    # excess = s - 1
    # We can replace one denominator d with two denominators d1, d2 such that 1/d = 1/d1 + 1/d2
    # This increases term count by 1, which we don't want.
    
    # Better: find 10 terms directly.
    # Let's try: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + x = 1
    # x = 1 - (1/5+...+1/13)
    s9 = sum(Fraction(1, d) for d in range(5, 14))
    x = 1 - s9
    if x > 0:
        # x = 1/d for some integer d?
        if x.denominator == 1:
            d = x.numerator
            if d >= 14 and d <= 10000000:
                sol = list(range(5, 14)) + [d]
                if len(set(sol)) == 10:
                    return sol
    
    # Try different combinations
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/14 + x = 1
    for skip in range(5, 14):
        s9 = sum(Fraction(1, d) for d in range(5, 14) if d != skip)
        x = 1 - s9
        if x > 0 and x.denominator == 1:
            d = x.numerator
            if d >= 5 and d != skip and d <= 10000000:
                sol = [d2 for d2 in range(5, 14) if d2 != skip] + [d]
                if len(set(sol)) == 10 and all(5 <= v <= 10000000 for v in sol):
                    s = sum(Fraction(1, v) for v in sol)
                    if s == 1:
                        return sorted(sol)
    
    # Fallback: return a known valid solution with more terms if needed
    # The seed program's approach: DP on 2520
    # Let's just return a valid solution. We know 10 terms is minimal.
    # Let's do a quick search for 10 terms with a different strategy
    
    # Greedy with backtracking for 10 terms
    def greedy_backtrack(k, remaining, min_d, used):
        if time.time() > deadline - 3.0:
            return None
        if k == 0:
            return [] if remaining == 0 else None
        if remaining <= 0:
            return None
        
        # Upper bound check
        ub = sum(Fraction(1, min_d + i) for i in range(k))
        if ub < remaining:
            return None
        
        max_d = int(k / float(remaining)) + 1
        max_d = min(max_d, 10000000)
        
        for d in range(max(min_d, 5), max_d + 1):
            if d in used:
                continue
            nr = remaining - Fraction(1, d)
            if nr < 0:
                break
            sub = greedy_backtrack(k - 1, nr, d + 1, used | {d})
            if sub is not None:
                return [d] + sub
        return None
    
    sol = greedy_backtrack(10, Fraction(1, 1), 5, set())
    if sol is not None:
        s = sum(Fraction(1, d) for d in sol)
        if s == 1 and len(set(sol)) == 10:
            return sorted(sol)
    
    # Final fallback: use the DP approach from the seed but with a larger target
    # Or just return a hardcoded valid solution
    # Let's construct one: 
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 - excess
    # excess = sum(1/5..1/14) - 1
    # We can adjust by replacing one term.
    
    # Actually, let's just do the DP from the seed with target 2520 as fallback
    target = 2520
    reached = {0: []}
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
