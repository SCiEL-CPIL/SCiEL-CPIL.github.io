"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_VAL = 1000000
    
    best_result = [int(d) for d in divisors]
    best_spread = max(best_result) - min(best_result)
    
    # Search over window size S from 9 upward
    # For each S, find if there's a valid L and assignment
    # Use periodicity with P = lcm(divisors) = 30030
    
    P = 30030
    
    for S in range(9, 150):
        if time.time() > deadline - 3:
            break
        
        # For this S, find valid L values in [0, P)
        # L is valid if for all i, there's a multiple of divisors[i] in [L, L+S]
        
        # Build valid L array
        valid = [True] * P
        
        for d in divisors:
            div_valid = [False] * P
            for k in range(P // d + 1):
                lo = max(0, k * d - S)
                hi = min(P - 1, k * d)
                if lo <= hi:
                    for l in range(lo, hi + 1):
                        div_valid[l] = True
            for l in range(P):
                if not div_valid[l]:
                    valid[l] = False
        
        # Find best L for this S
        found_better = False
        for L in range(P):
            if not valid[L]:
                continue
            
            # For this L, find assignment minimizing actual spread
            # For each divisor, possible values in [L, L+S]
            possible = []
            ok = True
            for d in divisors:
                kmin = (L + d - 1) // d
                kmax = (L + S) // d
                if kmin > kmax:
                    ok = False
                    break
                vals = [k * d for k in range(kmin, kmax + 1)]
                possible.append(vals)
            
            if not ok:
                continue
            
            # Pick distinct values minimizing spread
            # Greedy: try to cluster
            used = set()
            chosen = []
            for i in range(n):
                best_v = None
                for v in possible[i]:
                    if v not in used:
                        best_v = v
                        break
                if best_v is None:
                    # Try next value
                    found = False
                    for v in possible[i]:
                        if v not in used:
                            best_v = v
                            found = True
                            break
                    if not found:
                        break
                used.add(best_v)
                chosen.append(best_v)
            
            if len(chosen) < n:
                continue
            
            spread = max(chosen) - min(chosen)
            if spread < best_spread:
                best_spread = spread
                best_result = [int(v) for v in chosen]
                found_better = True
        
        if found_better:
            # Try to see if we can do even better with this S or smaller
            # Continue searching
            pass
    
    return best_result
# EVOLVE-BLOCK-END
