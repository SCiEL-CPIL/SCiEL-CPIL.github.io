import math
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55 seconds to be safe within 60s limit
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def verify_solution(denoms):
        """Verify that sum of reciprocals equals exactly 1."""
        if len(denoms) < 1 or len(denoms) > 256:
            return False
        if len(set(denoms)) != len(denoms):
            return False
        for d in denoms:
            if d < 5 or d > 10000000:
                return False
        s = Fraction(0, 1)
        for d in denoms:
            s += Fraction(1, d)
        return s == Fraction(1, 1)

    def greedy_egyptian(target_frac, min_denom, max_terms):
        """Greedy algorithm: repeatedly pick the largest unit fraction <= remaining."""
        remaining = target_frac
        chosen = []
        used_denoms = set()
        
        for _ in range(max_terms):
            if remaining <= 0:
                break
            if time.time() > deadline:
                return None
            
            if remaining.numerator == 0:
                break
                
            d_min_from_frac = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
            d_min = max(min_denom, d_min_from_frac)
            
            found = False
            for d in range(d_min, min(d_min + 100000, 10000001)):
                if d in used_denoms:
                    continue
                if d > 10000000:
                    break
                unit = Fraction(1, d)
                if unit <= remaining:
                    terms_left = max_terms - len(chosen) - 1
                    if terms_left > 0:
                        if remaining - unit > terms_left * Fraction(1, d + 1):
                            continue
                    
                    chosen.append(d)
                    used_denoms.add(d)
                    remaining -= unit
                    min_denom = d + 1
                    found = True
                    break
            
            if not found:
                return None
        
        if remaining == 0:
            return chosen
        return None

    def search_optimized(num, den, k, min_d, path, used_set, time_budget):
        """Optimized DFS with aggressive pruning for small k."""
        if time.time() > deadline:
            return None
        
        if k == 0:
            return path if num == 0 and den == 1 else None
            
        if k == 1:
            if den % num == 0:
                d = den // num
                if d >= min_d and d <= 10000000 and d not in used_set:
                    return path + [d]
            return None

        # Bounds for d
        d_min = max(min_d, math.ceil(den / num))
        # For k terms each >= 1/d, we need k/d >= num/den, so d <= k*den/num
        d_max = int((k * den) / num)
        
        if d_max > 10000000:
            d_max = 10000000
            
        if d_min > d_max:
            return None
        
        # Additional pruning: even with smallest possible denominators, can we reach num/den?
        # Sum of 1/(d_min), 1/(d_min+1), ..., 1/(d_min+k-1) must be >= num/den
        # Use a quick float check
        max_possible_sum = 0.0
        for i in range(k):
            d_candidate = d_min + i
            if d_candidate <= 10000000:
                max_possible_sum += 1.0 / d_candidate
        
        if max_possible_sum * den < num:
            return None
        
        # Pruning: even with largest possible denominators, can we not exceed num/den?
        # Sum of 1/(d_max-k+1), ..., 1/d_max must be <= num/den (rough check)
        # Actually we need sum >= num/den, so this is less useful
        
        # Iterate d from d_min to d_max
        # Try smaller d first (more likely to work)
        for d in range(d_min, d_max + 1):
            if d in used_set:
                continue
            
            # Check: 1/d <= num/den (i.e., d >= den/num)
            if d * num < den:
                continue
            
            # Check: remaining (num/den - 1/d) must be representable by k-1 terms with denominators > d
            # num/den - 1/d = (num*d - den) / (den*d)
            n2 = num * d - den
            d2 = den * d
            
            if n2 <= 0:
                continue
            
            # Quick float check: can k-1 terms with denominators >= d+1 sum to n2/d2?
            # Max sum: 1/(d+1) + 1/(d+2) + ... + 1/(d+k-1)
            max_rem_sum = 0.0
            for i in range(k - 1):
                d_rem = d + 1 + i
                if d_rem <= 10000000:
                    max_rem_sum += 1.0 / d_rem
            
            if max_rem_sum * d2 < n2:
                continue
            
            # Min sum: 1/(d+1) + ... but we need at least n2/d2
            # Actually we need sum = n2/d2, and each term <= 1/(d+1)
            # So (k-1)/(d+1) >= n2/d2 must hold
            if (k - 1) * d2 < n2 * (d + 1):
                continue
            
            g = gcd(n2, d2)
            n2_reduced = n2 // g
            d2_reduced = d2 // g
            
            used_set.add(d)
            result = search_optimized(n2_reduced, d2_reduced, k - 1, d + 1, path + [d], used_set, time_budget)
            used_set.remove(d)
            
            if result is not None:
                return result
        
        return None

    # Hardcoded known solutions - try shorter ones first
    # Verify these are valid: sum of reciprocals = 1, all in [5, 10000000], distinct
    hardcoded_candidates = [
        # Try to find 4-term solutions
        # 1/5 + 1/6 + 1/7 + 1/420 = (42+35+30+1)/420 = 108/420 = 9/35 ≠ 1
        # Known: 1/2+1/3+1/6=1 but 2,3 < 5
        # For min denom 5, minimal terms is likely >= 5
        # 5-term: 1/5+1/6+1/7+1/10+1/210 = ?
        # Let's just verify candidates
        [5, 6, 7, 10, 14, 20, 42, 70, 105, 210],
        [5, 6, 7, 10, 14, 21, 35, 42, 70, 105],
        [5, 6, 7, 10, 15, 20, 42, 70, 105, 210],
        [5, 6, 7, 10, 14, 35, 42, 70, 105, 210],
    ]
    
    best_solution = None
    best_k = 257
    
    for candidate in hardcoded_candidates:
        if verify_solution(candidate):
            if len(candidate) < best_k:
                best_solution = candidate
                best_k = len(candidate)
    
    # If no hardcoded solution works, use greedy
    if best_solution is None:
        best_solution = greedy_egyptian(Fraction(1, 1), 5, 256)
        if best_solution is not None:
            best_k = len(best_solution)
    
    # Try to find shorter solutions with optimized DFS
    # Start from k=4 and go up
    for k in range(4, min(best_k, 11)):
        if time.time() > deadline:
            break
            
        # Use a time budget for each k
        t_start = time.time()
        solution = search_optimized(1, 1, k, 5, [], set(), 10.0)
        t_elapsed = time.time() - t_start
        
        if solution is not None:
            if verify_solution(solution):
                best_solution = solution
                best_k = k
                break
        else:
            # If we spent too much time, stop
            if t_elapsed > 15.0:
                break
    
    # Final fallback if still no solution
    if best_solution is None:
        fallback = [5, 6, 7, 10, 14, 20, 42, 70, 105, 210]
        if verify_solution(fallback):
            best_solution = fallback
        else:
            best_solution = [5, 6, 7, 10, 14, 35, 42, 70, 105, 210]
    
    return best_solution
# EVOLVE-BLOCK-END
