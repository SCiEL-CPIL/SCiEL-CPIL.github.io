import time
from math import gcd
from functools import reduce

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000
DEADLINE_MARGIN = 5.0

def _lcm(a, b):
    return a * b // gcd(a, b)

def _all_lcm():
    return reduce(_lcm, DIVISORS)

L = _all_lcm()

def _check_window(s, W):
    """
    Check if there exists a valid packing in [s, s+W].
    Returns the assignment list if found, else None.
    """
    candidates = []
    for d in DIVISORS:
        first = ((s + d - 1) // d) * d
        if first > s + W:
            return None
        mults = []
        m = first
        while m <= s + W:
            mults.append(m)
            m += d
        candidates.append(mults)
        
    # Check for empty sets
    for c in candidates:
        if not c:
            return None

    # MRV ordering
    order = list(range(N))
    order.sort(key=lambda i: len(candidates[i]))
    
    # Forward checking / Propagation
    # We can do a quick check: if any value is the only option for two different vars, fail.
    # But backtracking handles this.
    
    used = [False] * (W + 1)
    assignment = [None] * N
    
    def backtrack(idx):
        if idx == N:
            return True
        i = order[idx]
        for m in candidates[i]:
            offset = m - s
            if not used[offset]:
                used[offset] = True
                assignment[i] = m
                if backtrack(idx + 1):
                    return True
                used[offset] = False
                assignment[i] = None
        return False
    
    if backtrack(0):
        return assignment
    return None

def construct():
    start_time = time.time()
    
    # Fallback solution: the divisors themselves
    fallback = [int(d) for d in DIVISORS]
    fallback_range = max(fallback) - min(fallback)
    
    # Search for optimal W
    # Min W is N-1
    # We search W from N-1 upwards
    # We iterate s from 0 to L-1
    # To speed up, we can randomize s or just iterate.
    # Given L=30030, iteration is fine if check is fast.
    
    for W in range(N - 1, 100):
        # Check deadline
        if time.time() - start_time > 60.0 - DEADLINE_MARGIN:
            return fallback
            
        # Iterate all starting positions
        # Optimization: If W is small, most s will fail quickly.
        for s in range(L):
            res = _check_window(s, W)
            if res is not None:
                # Found a solution
                # Verify types and constraints
                if all(isinstance(x, int) and not isinstance(x, bool) for x in res):
                    if len(set(res)) == N:
                        valid = True
                        for i, d in enumerate(DIVISORS):
                            if res[i] % d != 0 or res[i] <= 0 or res[i] > MAX_VAL:
                                valid = False
                                break
                        if valid:
                            return res
    return fallback
