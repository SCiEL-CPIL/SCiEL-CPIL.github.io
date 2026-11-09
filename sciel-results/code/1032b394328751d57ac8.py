import time
from fractions import Fraction

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

    # First, let's find a valid solution using a systematic approach
    # We'll use the greedy Egyptian fraction algorithm as a baseline
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

    # Try to find a valid solution with the greedy algorithm
    greedy_solution = greedy_egyptian(Fraction(1, 1), 5, 256)
    
    # Now let's try to find a shorter solution using DFS
    def search_optimized(num, den, k, min_d, path, used_set):
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
        d_min = max(min_d, (den + num - 1) // num)
        # For k terms each >= 1/d, we need k/d >= num/den, so d <= k*den/num
        d_max = min(int((k * den) // num), 10000000)
        
        if d_min > d_max:
            return None
        
        # Iterate d from d_min to d_max
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
            
            # Min sum check: (k-1)/(d+1) >= n2/d2 must hold
            if (k - 1) * d2 < n2 * (d + 1):
                continue
            
            g = gcd(n2, d2)
            n2_reduced = n2 // g
            d2_reduced = d2 // g
            
            used_set.add(d)
            result = search_optimized(n2_reduced, d2_reduced, k - 1, d + 1, path + [d], used_set)
            used_set.remove(d)
            
            if result is not None:
                return result
        
        return None

    # Start with the greedy solution as our best
    best_solution = greedy_solution
    best_k = len(greedy_solution) if greedy_solution else 257

    # Try to find shorter solutions with optimized DFS
    # Start from k=5 and go up (k=4 is likely impossible with min denom 5)
    for k in range(5, min(best_k, 12)):
        if time.time() > deadline:
            break
            
        # Use a time budget for each k
        t_start = time.time()
        solution = search_optimized(1, 1, k, 5, [], set())
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
        # Use greedy as final fallback
        best_solution = greedy_egyptian(Fraction(1, 1), 5, 256)
    
    return best_solution
