"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import functools

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    period = functools.reduce(lcm, divisors, 1)  # 30030
    
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    best_result = None
    best_range = float('inf')
    
    for R in range(9, 500):  # Try window sizes from 9 to 499
        if time.time() > deadline:
            break
        
        found = False
        
        for L in range(period):
            if time.time() > deadline:
                break
            
            # Quick feasibility: each divisor must have at least one multiple in [L, L+R]
            candidates = []
            valid = True
            
            for d in divisors:
                k_min = (L + d - 1) // d
                k_max = (L + R) // d
                
                if k_max < k_min:
                    valid = False
                    break
                
                cands = [k * d for k in range(k_min, k_max + 1)]
                candidates.append(cands)
            
            if not valid:
                continue
            
            # Backtracking with MRV
            indices = sorted(range(n), key=lambda i: len(candidates[i]))
            
            used = set()
            assignment = [None] * n
            
            def backtrack(idx):
                if idx == n:
                    return True
                d_idx = indices[idx]
                for m in candidates[d_idx]:
                    if m not in used:
                        used.add(m)
                        assignment[d_idx] = m
                        if backtrack(idx + 1):
                            return True
                        used.remove(m)
                        assignment[d_idx] = None
                return False
            
            if backtrack(0):
                actual_vals = [assignment[i] for i in range(n)]
                actual_range = max(actual_vals) - min(actual_vals)
                
                if actual_range < best_range:
                    best_range = actual_range
                    best_result = [int(v) for v in actual_vals]
                
                found = True
                break
        
        if found:
            return best_result
    
    if best_result is None:
        return [int(d) for d in divisors]
    
    return best_result

# EVOLVE-BLOCK-END
