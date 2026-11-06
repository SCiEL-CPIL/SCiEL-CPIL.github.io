"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    def verify(sol):
        if not sol:
            return False
        if len(set(sol)) != len(sol):
            return False
        if any(d < 5 or d > 10000000 for d in sol):
            return False
        # Use exact fraction arithmetic
        num, den = 0, 1
        for d in sol:
            new_num = num * d + den
            new_den = den * d
            g = gcd(new_num, new_den)
            num = new_num // g
            den = new_den // g
        return num == den
    
    def _search(num, den, k, min_d, path):
        """Find k distinct denominators >= min_d such that sum of 1/d = num/den"""
        if time.time() > deadline:
            return None
        if k == 0:
            if num == den:
                return path[:]
            return None
        if k == 1:
            # Need 1/d = num/den => d = den/num
            if den % num == 0:
                d = den // num
                if d >= min_d:
                    if not path or d > path[-1]:
                        return path + [d]
            return None
        
        # Pruning: maximum possible sum with k terms starting from min_d
        # is 1/min_d + 1/(min_d+1) + ... + 1/(min_d+k-1)
        max_num, max_den = 0, 1
        for i in range(k):
            d = min_d + i
            new_num = max_num * d + max_den
            new_den = max_den * d
            g = gcd(new_num, new_den)
            max_num = new_num // g
            max_den = new_den // g
        if max_num * den < num * max_den:
            return None
        
        # Lower bound on d: 1/d <= num/den => d >= den/num
        if num == 0:
            return None
        d_start = max(min_d, den // num + (1 if den % num == 0 else 0))
        if path:
            d_start = max(d_start, path[-1] + 1)
        
        # Upper bound: 1/d + (k-1)/(d+1) >= num/den
        # For a rough upper bound: k/d >= num/den => d <= k*den/num
        d_upper = min(10000000, (k * den) // num + k + 20)
        
        for d in range(d_start, d_upper + 1):
            if time.time() > deadline:
                return None
            # new remainder: num/den - 1/d = (d*num - den)/(d*den)
            new_num = d * num - den
            if new_num <= 0:
                continue
            new_den = d * den
            g = gcd(new_num, new_den)
            rn = new_num // g
            rd = new_den // g
            result = _search(rn, rd, k - 1, d + 1, path + [d])
            if result is not None:
                return result
        
        return None
    
    # Try increasing k from 8
    for k in range(8, 13):
        sol = _search(1, 1, k, 5, [])
        if sol is not None and verify(sol):
            return sol
    
    # Fallback: verified 15-term solution
    fallback = [5, 6, 7, 10, 12, 14, 15, 21, 30, 35, 42, 70, 105, 140, 210]
    if verify(fallback):
        return fallback
    
    return fallback
# EVOLVE-BLOCK-END
