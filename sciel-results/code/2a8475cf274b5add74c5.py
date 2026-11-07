import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
START_TIME = time.time()
TIME_LIMIT = 50.0  # Safety margin for 60s limit

def lcm(a, b):
    return a * b // gcd(a, b)

LCM_ALL = reduce(lcm, DIVISORS)

def check_and_solve(L, R):
    """
    Check if there exist distinct multiples x_i of DIVISORS[i] in [L, L+R].
    Returns list of x_i if found, else None.
    """
    candidates = []
    for i, d in enumerate(DIVISORS):
        # First multiple >= L
        m = ((L + d - 1) // d) * d
        if m > L + R:
            return None
        # Generate all multiples in [L, L+R]
        vals = []
        curr = m
        while curr <= L + R:
            vals.append(curr)
            curr += d
        candidates.append(vals)
    
    # Backtracking to find distinct assignment
    # Sort by number of candidates (most constrained first)
    # We need to keep track of original indices
    indices = list(range(len(DIVISORS)))
    indices.sort(key=lambda i: len(candidates[i]))
    
    assigned = {}
    used_vals = set()
    
    def backtrack(idx):
        if idx == len(indices):
            return True
        i = indices[idx]
        for v in candidates[i]:
            if v not in used_vals:
                used_vals.add(v)
                assigned[i] = v
                if backtrack(idx + 1):
                    return True
                used_vals.remove(v)
        return False
    
    if backtrack(0):
        result = [0] * len(DIVISORS)
        for i in range(len(DIVISORS)):
            result[i] = assigned[i]
        return result
    return None

def construct():
    # Fallback solution: the divisors themselves
    # They are distinct and positive.
    # Range: 39 - 6 = 33.
    fallback = DIVISORS[:]
    
    # Search for optimal R
    # R must be at least 1 because distinct integers
    for R in range(1, 1000):
        if time.time() - START_TIME > TIME_LIMIT:
            return fallback
            
        for L in range(1, LCM_ALL + 1):
            if time.time() - START_TIME > TIME_LIMIT:
                return fallback
                
            res = check_and_solve(L, R)
            if res is not None:
                # Verify distinctness and bounds
                if len(set(res)) == len(res) and all(1 <= x <= MAX_VAL for x in res):
                    return res
                    
    return fallback
# EVOLVE-BLOCK-END
