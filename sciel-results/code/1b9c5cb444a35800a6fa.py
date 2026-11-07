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
    
    # For W < min(divisors) = 6, each divisor has at most one candidate in [L, L+W]
    # Try W = 1, 2, 3, 4, 5
    for W in range(1, 6):
        if time.time() > deadline:
            break
        
        found = False
        for L in range(1, period + 1):
            if time.time() > deadline:
                break
            
            R = L + W
            if R > max_val:
                break
            
            vals = []
            valid = True
            for d in divisors:
                rem = L % d
                m = L if rem == 0 else L + (d - rem)
                if m > R:
                    valid = False
                    break
                vals.append(m)
            
            if not valid:
                continue
            
            if len(set(vals)) != n:
                continue
            
            found = True
            diff = max(vals) - min(vals)
            if diff < best_diff:
                best_diff = diff
                best_solution = vals[:]
                if best_diff == 1:
                    return [int(x) for x in best_solution]
            break
        
        if found:
            break
    
    if best_solution is not None:
        return [int(x) for x in best_solution]
    
    # For W >= 6, use a sliding window approach with backtracking
    # Key optimization: only check L values that are "interesting" (near multiples)
    # But simpler: limit L range and use faster candidate generation
    
    def try_window(L, W):
        """Try to find distinct assignment in [L, L+W]. Returns assignment or None."""
        R = L + W
        candidates = []
        
        for i, d in enumerate(divisors):
            k = (L + d - 1) // d
            x = k * d
            cands = []
            while x <= R:
                cands.append(x)
                x += d
            candidates.append(cands)
        
        # Sort indices by candidate count (most constrained first)
        order = sorted(range(n), key=lambda i: len(candidates[i]))
        
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
    
    # Try W from 6 upward, but limit L search to a smaller range
    # Since the problem is periodic with 'period', we only need to check L in [1, period]
    # But we can be smarter: for each W, only check L where at least one new multiple enters
    # For now, just limit L range to speed things up
    max_L_check = min(period, 5000)  # Limit L checks for speed
    
    for W in range(6, 100):
        if time.time() > deadline:
            break
        
        found = False
        for L in range(1, max_L_check + 1):
            if time.time() > deadline:
                break
            
            R = L + W
            if R > max_val:
                break
            
            sol = try_window(L, W)
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
