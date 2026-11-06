"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    # Known solutions with small denominators - verify and return immediately
    known_solutions = [
        [5, 6, 7, 42, 140, 210, 420, 840, 1260, 2520],
        [5, 7, 10, 35, 70, 140, 210, 420, 840, 1680],
        [6, 7, 10, 15, 21, 35, 42, 70, 105, 210],
    ]
    
    for sol in known_solutions:
        s = sum(Fraction(1, d) for d in sol)
        if s == 1 and len(set(sol)) == len(sol) and all(5 <= d <= 10000000 for d in sol):
            return sol

    # Try to find a 9-term solution using targeted DFS with integer arithmetic
    def dfs(rem_num, rem_den, min_d, terms_left, current):
        if time.time() > deadline:
            return None
        if terms_left == 0:
            return None if rem_num != 0 else []
        
        # Each term <= 1/min_d, so terms_left/min_d >= rem_num/rem_den
        if terms_left * rem_den < rem_num * min_d:
            return None
        
        # d must satisfy: 1/d <= rem_num/rem_den => d >= rem_den/rem_num
        d_lower = max(min_d, -(-rem_den // rem_num))
        
        # d upper bound: rem_num/d - 1 > 0 => d < rem_den*rem_num/rem_num... 
        # Actually: rem_num/rem_den - 1/d > 0 => rem_num*d > rem_den => d > rem_den/rem_num (have this)
        # For remaining terms to be achievable: rem_num/rem_den - 1/d <= (terms_left-1)/(d+1)
        # => (rem_num*d - rem_den)/(rem_den*d) <= (terms_left-1)/(d+1)
        # => (rem_num*d - rem_den)*(d+1) <= (terms_left-1)*rem_den*d
        # This is quadratic; let's use a loose upper bound
        d_upper = min(10000000, rem_den * terms_left + 100)
        
        for d in range(d_lower, d_upper + 1):
            if time.time() > deadline:
                return None
            
            if rem_num * d <= rem_den:
                continue
            
            new_num = rem_num * d - rem_den
            new_den = rem_den * d
            g = gcd(new_num, new_den)
            new_num //= g
            new_den //= g
            
            # Prune: max possible sum of (terms_left-1) terms starting at d+1
            if (terms_left - 1) * (d + 1) < new_num * new_den:
                break
            
            # Prune: min possible sum
            if (terms_left - 1) * new_den < new_num * 10000000:
                continue
            
            result = dfs(new_num, new_den, d + 1, terms_left - 1, current + [d])
            if result is not None:
                return current + [d] + result
        
        return None

    # Try 9 terms
    if time.time() < deadline - 10:
        result = dfs(1, 1, 5, 9, [])
        if result is not None:
            from fractions import Fraction
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result

    # Fallback: use a known representation
    for sol in known_solutions:
        from fractions import Fraction
        s = sum(Fraction(1, d) for d in sol)
        if s == 1 and len(set(sol)) == len(sol) and all(5 <= d <= 10000000 for d in sol):
            return sol

    raise RuntimeError("no solution found")
# EVOLVE-BLOCK-END
