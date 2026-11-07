import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin within 60s limit
    
    divisors = P['divisors']
    n = len(divisors)
    max_val = P['maximum']
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    period = reduce(lcm, divisors)  # LCM of all divisors
    
    best_diff = float('inf')
    best_solution = None
    
    # Key insight: For W < min(divisors), each divisor has at most ONE candidate in [L, L+W]
    # So we just need to check if all 10 computed multiples are distinct
    # min(divisors) = 6, so for W < 6, this is guaranteed unique
    
    # First, try small W values where uniqueness is guaranteed (W < 6)
    # For each such W, slide L over one full period
    for W in range(1, 6):  # W = 1, 2, 3, 4, 5
        if time.time() > deadline:
            break
        
        found = False
        for L in range(1, period + 1):
            if time.time() > deadline:
                break
            
            R = L + W
            if R > max_val:
                break
            
            # For each divisor, compute the unique multiple in [L, R] if it exists
            vals = []
            valid = True
            for d in divisors:
                # Smallest multiple of d >= L
                if L % d == 0:
                    m = L
                else:
                    m = L + (d - L % d)
                if m > R:
                    valid = False
                    break
                vals.append(m)
            
            if not valid:
                continue
            
            # Check distinctness
            if len(set(vals)) != n:
                continue
            
            # Valid solution found
            found = True
            diff = max(vals) - min(vals)
            if diff < best_diff:
                best_diff = diff
                best_solution = vals[:]
                if best_diff == 1:
                    return [int(x) for x in best_solution]
            break  # Found solution for this W, no need to continue for this W
        
        if found:
            break
    
    if best_solution is not None:
        return [int(x) for x in best_solution]
    
    # If no solution found for W < 6, try larger W with backtracking
    # But first, let's try a more systematic approach for W = 6, 7, ...
    
    def check_with_backtrack(L, W):
        """Check if a valid distinct assignment exists in [L, L+W] using backtracking."""
        R = L + W
        candidates = [[] for _ in range(n)]
        
        for i, d in enumerate(divisors):
            k = (L + d - 1) // d
            x = k * d
            while x <= R:
                candidates[i].append(x)
                x += d
        
        # Sort by number of candidates (most constrained first)
        order = sorted(range(n), key=lambda i: len(candidates[i]))
        
        # Quick check: if any has 0 candidates, fail
        if len(candidates[order[0]]) == 0:
            return None
        
        used = set()
        assignment = [0] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            i = order[idx]
            for val in candidates[i]:
                if val not in used:
                    used.add(val)
                    assignment[i] = val
                    if backtrack(idx + 1):
                        return True
                    used.remove(val)
            return False
        
        if backtrack(0):
            return assignment
        return None
    
    # Try W from 6 upward with backtracking
    for W in range(6, 200):
        if time.time() > deadline:
            break
        
        found = False
        for L in range(1, period + 1):
            if time.time() > deadline:
                break
            
            R = L + W
            if R > max_val:
                break
            
            sol = check_with_backtrack(L, W)
            if sol is not None:
                found = True
                diff = max(sol) - min(sol)
                if diff < best_diff:
                    best_diff = diff
                    best_solution = sol[:]
                break
        
        if found:
            break
    
    if best_solution is not None:
        return [int(x) for x in best_solution]
    
    # Fallback: return the divisors themselves (valid but not optimal)
    return [int(d) for d in divisors]

# EVOLVE-BLOCK-END
