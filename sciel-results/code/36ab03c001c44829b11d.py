import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    deadline = time.time() + 55.0
    
    lcm_val = reduce(_lcm, divisors)
    
    best_spread = float('inf')
    best_values = None
    
    # For a given window [L, L+W], check if we can assign distinct multiples
    # using backtracking with bitsets for efficiency
    def check_assignment(L, W):
        # For each divisor, find candidate multiples in [L, L+W]
        cands = []
        for d in divisors:
            start = ((L + d - 1) // d) * d
            if start > L + W:
                return None
            c = []
            x = start
            while x <= L + W:
                c.append(x)
                x += d
            cands.append(c)
            
        if any(len(c) == 0 for c in cands):
            return None
            
        # Order by number of candidates (most constrained first)
        order = sorted(range(n), key=lambda i: len(cands[i]))
        
        # Use a set for tracking used values
        used = set()
        assigned = [None] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            orig_idx = order[idx]
            for val in cands[orig_idx]:
                if val not in used:
                    used.add(val)
                    assigned[orig_idx] = val
                    if backtrack(idx + 1):
                        return True
                    used.remove(val)
            return False
            
        if backtrack(0):
            return assigned
        return None

    # Search strategy:
    # 1. First try small W values (0 to 20) across all L positions
    # 2. If not found, expand W and continue searching
    # 3. Use periodicity: only need to check L in [1, lcm_val]
    
    # Precompute for each W, which L values are feasible
    # Start with small W values
    for W in range(0, 25):
        if time.time() > deadline - 3.0:
            break
            
        found = False
        # Check all possible L positions
        for L in range(1, lcm_val + 1):
            if time.time() > deadline - 3.0:
                break
                
            res = check_assignment(L, W)
            if res is not None:
                best_spread = W
                best_values = res
                found = True
                break
        
        if found:
            break
    
    # If not found with small W, do a more thorough search
    if best_values is None or best_spread > 15:
        # Binary search on W for each L
        for L in range(1, lcm_val + 1):
            if time.time() > deadline - 2.0:
                break
                
            lo = 0
            hi = 100
            
            # Check if hi is valid
            res_hi = check_assignment(L, hi)
            if res_hi is None:
                continue
                
            # Binary search for minimal W
            while lo < hi:
                mid = (lo + hi) // 2
                res_mid = check_assignment(L, mid)
                if res_mid is not None:
                    hi = mid
                else:
                    lo = mid + 1
                    
            if lo < best_spread:
                best_spread = lo
                res_best = check_assignment(L, lo)
                if res_best:
                    best_values = res_best
                    
            if best_spread == 9:
                break
    
    if best_values is None:
        return [int(d) for d in divisors]
        
    return [int(v) for v in best_values]
