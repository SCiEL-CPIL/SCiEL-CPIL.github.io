"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Try to find a small solution via bounded DFS
    # We want sum of 1/d_i = 1, all d_i in [5, 10000000], distinct, minimize count
    
    def dfs(remaining_num, remaining_den, min_d, used, current, max_terms):
        """
        remaining_num / remaining_den is the remaining sum to achieve.
        min_d: minimum allowed denominator (must be > all in current to ensure distinct and ordered)
        used: set of denominators used
        current: list of denominators chosen so far
        max_terms: maximum additional terms allowed
        Returns list of denominators to complete, or None
        """
        if time.time() > deadline:
            return None
        
        if remaining_num == 0:
            return []
        
        if max_terms == 0:
            return None
        
        # The next denominator d must satisfy: 1/d <= remaining_num/remaining_den
        # So d >= remaining_den / remaining_num
        min_d = max(min_d, (remaining_den + remaining_num - 1) // remaining_num)
        if min_d < 5:
            min_d = 5
        
        # Also, we need at least (max_terms) terms to sum to remaining_num/remaining_den
        # Each term <= 1/min_d, so max_terms * (1/min_d) >= remaining_num/remaining_den
        # => max_terms >= remaining_num * min_d / remaining_den
        if max_terms * remaining_den < remaining_num * min_d:
            return None
        
        # Try denominators starting from min_d
        # Limit the range to keep search tractable
        max_d_try = min(10000000, min_d + 5000)  # Only try nearby values
        
        d = min_d
        while d <= max_d_try:
            if time.time() > deadline:
                return None
            
            if d not in used:
                # New remaining: remaining_num/remaining_den - 1/d
                new_num = remaining_num * d - remaining_den
                new_den = remaining_den * d
                
                if new_num >= 0:
                    if new_num == 0:
                        return [d]
                    
                    # Reduce fraction
                    g = math.gcd(new_num, new_den)
                    if g > 1:
                        new_num //= g
                        new_den //= g
                    
                    # Pruning: can we still complete with remaining terms?
                    # Each remaining term <= 1/(d+1), so (max_terms-1) * (1/(d+1)) >= new_num/new_den
                    if (max_terms - 1) * new_den >= new_num * (d + 1):
                        new_used = used | {d}
                        result = dfs(new_num, new_den, d + 1, new_used, current + [d], max_terms - 1)
                        if result is not None:
                            return [d] + result
            
            d += 1
        
        return None
    
    # Try small number of terms first
    for max_k in range(3, 10):
        if time.time() > deadline:
            break
        result = dfs(1, 1, 5, set(), [], max_k)
        if result is not None:
            if 1 <= len(result) <= 256:
                # Verify
                s = sum(Fraction(1, x) for x in result)
                if s == 1:
                    return result
    
    # Fallback: Greedy algorithm
    num, den = 1, 1
    used = set()
    result = []
    
    while num > 0:
        if time.time() > deadline:
            break
            
        min_d = (den + num - 1) // num
        if min_d < 5:
            min_d = 5
            
        d = min_d
        while d in used:
            d += 1
            
        if d > 10000000:
            break
            
        result.append(d)
        used.add(d)
        
        num = num * d - den
        den = den * d
        
        g = math.gcd(num, den)
        num //= g
        den //= g
    
    if num == 0 and len(result) > 0 and len(result) <= 256:
        # Local search: try to reduce count by merging pairs
        def try_merge(result, used):
            improved = True
            while improved:
                improved = False
                n = len(result)
                for i in range(n):
                    if time.time() > deadline:
                        return result
                    for j in range(i+1, n):
                        a, b = result[i], result[j]
                        num = a * b
                        den = a + b
                        if den > 0 and num % den == 0:
                            c = num // den
                            if c >= 5 and c <= 10000000 and c not in used:
                                new_result = []
                                new_used = set()
                                for k in range(n):
                                    if k != i and k != j:
                                        new_result.append(result[k])
                                        new_used.add(result[k])
                                new_result.append(c)
                                new_used.add(c)
                                s = sum(Fraction(1, x) for x in new_result)
                                if s == 1 and len(new_result) < len(result):
                                    result = new_result
                                    used = new_used
                                    improved = True
                                    break
                    if improved:
                        break
            return result
        
        result = try_merge(result, used)
        
        s = sum(Fraction(1, x) for x in result)
        if s == 1 and 1 <= len(result) <= 256:
            return result
    
    # Final fallback: DP over divisors of 2520
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
