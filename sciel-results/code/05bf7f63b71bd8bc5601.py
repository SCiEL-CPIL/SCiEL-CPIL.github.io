import time
from math import gcd
from functools import reduce

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    LCM = reduce(lcm, divisors)  # 30030
    
    start_time = time.time()
    timeout = 85  # Safety margin for 90s limit
    
    # We search for minimal W (max - min)
    # Minimum W is 9 (10 distinct integers)
    # We will iterate W from 9 upwards
    for W in range(9, 300):
        # Check all possible starting positions L modulo LCM
        # L ranges from 1 to LCM
        for L in range(1, LCM + 1):
            if time.time() - start_time > timeout:
                # Fallback: return a valid solution
                # The divisors themselves are distinct and valid, range 39-6=33
                return divisors[:]
            
            # For this window [L, L+W], determine candidates for each divisor
            # forced[i] will hold the candidate if only 1 exists, None if >1 or 0
            # flexible[i] will hold (index, [cand1, cand2]) if 2 exist
            # If 0 exist, we break early
            forced = [None] * n
            flexible = []
            valid_window = True
            
            for i, d in enumerate(divisors):
                # Find first multiple >= L
                k = (L + d - 1) // d
                c1 = k * d
                
                if c1 > L + W:
                    valid_window = False
                    break
                
                c2 = c1 + d
                if c2 > L + W:
                    # Only c1 is available
                    forced[i] = c1
                else:
                    # c1 and c2 are available
                    # (c2+d would be > L+W because W is small relative to d? 
                    # Actually, if W >= 2*d, there could be 3.
                    # But for W < 40, and min d=6, max d=39.
                    # If d=6, W=10, c1, c1+6, c1+12. 12 <= 10? No.
                    # So for W < 12, max 2 candidates for d=6.
                    # For W < 20, max 3 candidates for d=6.
                    # Let's handle generally: collect all candidates
                    candidates = []
                    x = c1
                    while x <= L + W:
                        candidates.append(x)
                        x += d
                    if len(candidates) == 1:
                        forced[i] = candidates[0]
                    else:
                        flexible.append((i, candidates))
            
            if not valid_window:
                continue
            
            # Check for conflicts in forced assignments
            used_forced = set()
            has_conflict = False
            for i in range(n):
                if forced[i] is not None:
                    if forced[i] in used_forced:
                        has_conflict = True
                        break
                    used_forced.add(forced[i])
            
            if has_conflict:
                continue
            
            # Now we need to assign values to flexible divisors
            # such that they are distinct and not in used_forced
            # Sort flexible by number of candidates (most constrained first)
            flexible.sort(key=lambda x: len(x[1]))
            
            # Backtracking search
            # State: index in flexible list
            # We need to store the final result in original indices
            result = [0] * n
            # Pre-fill forced
            for i in range(n):
                if forced[i] is not None:
                    result[i] = forced[i]
            
            used = set(used_forced)
            
            def backtrack(f_idx):
                if f_idx == len(flexible):
                    return True
                idx, candidates = flexible[f_idx]
                for val in candidates:
                    if val not in used:
                        used.add(val)
                        result[idx] = val
                        if backtrack(f_idx + 1):
                            return True
                        used.remove(val)
                return False
            
            if backtrack(0):
                # Verify just in case
                if len(set(result)) == n:
                    for i in range(n):
                        if result[i] % divisors[i] != 0 or not (0 < result[i] <= 1000000):
                            continue
                    return result
    
    # Fallback if no solution found in time or range
    # Return the divisors themselves (valid, range 33)
    return divisors[:]
