import time
from fractions import Fraction
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # wall-clock seconds; evaluator kills at 90s

    def elapsed():
        return time.time() - start_time

    def deadline_reached():
        return elapsed() > DEADLINE

    # We need distinct denominators in [5, 10^7] summing to 1.
    # Minimize count.
    
    # Strategy:
    # 1. Try to find a solution with 11 terms (or fewer) using a targeted search.
    # 2. If not found quickly, fall back to a known 12-term or 13-term solution.
    
    # Let's first check if a small number of terms is possible.
    # The minimum number of terms is at least 5 (since 1/5+1/5+1/5+1/5+1/5 = 1, but distinctness required).
    # With distinct denominators >= 5, the sum of the first 5 is 1/5+1/6+1/7+1/8+1/9 ≈ 0.2+0.166+0.142+0.125+0.111 = 0.744 < 1.
    # So we need more terms.
    
    # Let's use a backtracking search with strong pruning to find a solution with N terms.
    # We'll start with N=11, then 12, etc.
    
    def find_solution(n_terms):
        # We want to find d1 < d2 < ... < dn such that sum(1/di) = 1
        # d1 >= 5
        
        # Pruning bounds:
        # If we have chosen k terms, remaining sum is R.
        # We need to pick n_terms - k more terms, all > last_chosen.
        # Max possible sum of remaining terms: 1/(last+1) + 1/(last+2) + ... 
        # But a tighter bound: the sum of the first (n_terms-k) terms starting from last+1.
        # If R > that sum, impossible.
        # Min possible sum: 1/10^7 + ... (very small)
        # Also, 1/d <= R, so d >= 1/R.
        
        def backtrack(rem, min_d, chosen, k):
            if deadline_reached():
                return None
            if k == 0:
                if rem == 0:
                    return chosen[:]
                return None
            if rem <= 0:
                return None
            
            # Lower bound for d:
            # 1/d <= rem  =>  d >= 1/rem
            # Also d >= min_d
            d_min = max(min_d, math.ceil(1 / rem))
            
            # Upper bound for d:
            # The sum of the remaining k terms must be at least rem.
            # The maximum sum of k terms with denominators >= d is 1/d + 1/(d+1) + ... + 1/(d+k-1)
            # If rem > max_sum, then no solution.
            # We can approximate max_sum <= k/d.
            # So k/d >= rem  =>  d <= k/rem.
            d_max = min(10000000, int(k / rem))
            
            # Also, the sum of the next k terms starting from d_min must be at least rem.
            # Let's check if it's even possible.
            # Sum_{i=0}^{k-1} 1/(d_min + i)
            # If this sum < rem, then we need smaller d, but d_min is already the lower bound.
            # So if sum < rem, return None.
            
            # For efficiency, we can just iterate d from d_min to d_max.
            # But d_max might be large.
            
            # A better upper bound:
            # We need to be able to form rem from k terms.
            # The largest term is 1/d.
            # The remaining k-1 terms must sum to rem - 1/d.
            # The maximum sum of k-1 terms with denominators > d is (k-1)/(d+1).
            # So rem - 1/d <= (k-1)/(d+1)
            # rem <= 1/d + (k-1)/(d+1)
            # This gives an upper bound on d.
            
            # Let's solve for d:
            # rem <= 1/d + (k-1)/(d+1)
            # rem * d * (d+1) <= d + 1 + d * (k-1)
            # rem * d^2 + rem * d <= d + 1 + kd - k
            # rem * d^2 + (rem - 1 - k) * d + (k - 1) <= 0
            
            # This is a quadratic in d. Let's find the positive root.
            # a = rem, b = rem - 1 - k, c = k - 1
            a = rem
            b = rem - 1 - k
            c = k - 1
            disc = b*b - 4*a*c
            if disc < 0:
                # No real root, quadratic is always positive (since a>0), so no solution?
                # Wait, if a>0 and disc<0, the quadratic is always positive.
                # We want <= 0, so no solution.
                return None
            
            sqrt_disc = math.sqrt(disc)
            # Roots are (-b ± sqrt_disc) / (2a)
            # We want the larger root (since d must be positive and large enough).
            d_max_quad = int((-b + sqrt_disc) / (2 * a)) + 1
            
            d_max = min(d_max, d_max_quad)
            
            if d_min > d_max:
                return None
            
            # Iterate d from d_min to d_max
            # To speed up, we can limit the range if it's too large.
            # But let's try a reasonable range first.
            
            # Optimization: if the range is huge, we might miss solutions.
            # But for small k, the range should be manageable.
            
            for d in range(d_min, d_max + 1):
                if deadline_reached():
                    return None
                
                new_rem = rem - 1/d
                chosen.append(d)
                res = backtrack(new_rem, d + 1, chosen, k - 1)
                if res is not None:
                    return res
                chosen.pop()
            
            return None
        
        # Start with rem = 1, min_d = 5, chosen = [], k = n_terms
        return backtrack(1.0, 5, [], n_terms)
    
    # Try to find solutions with increasing number of terms
    # Start with 11, then 12, etc.
    # But 11 might take too long.
    # Let's try a few.
    
    # Actually, the known minimum for distinct denominators >= 5 is likely around 10-12.
    # Let's try 10 first, but with a time limit.
    
    # To make it faster, we can use a heuristic search or a different algorithm.
    # For now, let's try backtracking for 10 and 11 terms.
    
    # But backtracking might be too slow for 10 terms.
    # Let's use a greedy approach or a known result.
    
    # Known result:
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 + 1/42 + 1/70 + 1/105 + 1/210 + 1/420
    # Let's verify this sum.
    
    candidate = [5, 6, 7, 10, 14, 35, 42, 70, 105, 210, 420]
    s = sum(Fraction(1, d) for d in candidate)
    if s == 1:
        return candidate
    
    # If not, try another known decomposition.
    # 1 = 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 ...
    # This is getting too many terms.
    
    # Let's try to find a solution with 12 terms using backtracking, but with a tighter time limit.
    # Or just use a precomputed solution.
    
    # Precomputed 12-term solution:
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 ... no, sum is > 1.
    
    # Let's use the one from the seed program or a similar one.
    # The seed program used a fallback that found a 12-term solution.
    
    # Let's try to find a 12-term solution quickly.
    sol = find_solution(12)
    if sol:
        # Verify
        s = sum(Fraction(1, d) for d in sol)
        if s == 1 and len(set(sol)) == len(sol) and all(5 <= d <= 10000000 for d in sol):
            return sol
    
    # If not found, try 13 terms
    sol = find_solution(13)
    if sol:
        s = sum(Fraction(1, d) for d in sol)
        if s == 1 and len(set(sol)) == len(sol) and all(5 <= d <= 10000000 for d in sol):
            return sol
    
    # Fallback: use a known 12-term solution
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 + 1/42 + 1/70 + 1/105 + 1/210 + 1/420 + 1/840
    # Let's verify this.
    candidate = [5, 6, 7, 10, 14, 35, 42, 70, 105, 210, 420, 840]
    s = sum(Fraction(1, d) for d in candidate)
    if s == 1:
        return candidate
    
    # If all else fails, return a valid but possibly suboptimal solution
    # We can construct one by splitting 1/2 into fractions with denom >= 5.
    # 1/2 = 1/5 + 1/6 + 1/12 + 1/20? No.
    # 1/2 = 1/6 + 1/7 + 1/14 + 1/21 + 1/42?
    # 1/6 + 1/7 + 1/14 + 1/21 + 1/42 = 7/42 + 6/42 + 3/42 + 2/42 + 1/42 = 19/42 != 1/2.
    
    # Let's just use a large set of denominators that sum to 1.
    # For example, 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20
    # Sum is approx 0.2+0.166+0.142+0.125+0.111+0.1+0.09+0.083+0.076+0.071+0.066+0.0625+0.058+0.055+0.052+0.05 = 1.447 > 1.
    # So we need to adjust.
    
    # A safer fallback:
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30
    # This is too many terms and sum is way over 1.
    
    # Let's use the solution from the seed program's fallback.
    # The seed program used a DP-like approach with target 2520.
    
    def get_fallback():
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
                sol = reached[target]
                s = sum(Fraction(1, d) for d in sol)
                if s == 1 and len(set(sol)) == len(sol) and all(5 <= d <= 10000000 for d in sol):
                    return sol
        raise RuntimeError("no baseline representation")
    
    return get_fallback()
