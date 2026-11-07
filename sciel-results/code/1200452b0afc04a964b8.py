import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    divisors = DIVISORS
    n = N
    
    # Calculate LCM of all divisors to determine the period
    period = reduce(_lcm, divisors)
    
    # We search for the minimal span W.
    # The span is max(reps) - min(reps).
    # We iterate W from 1 upwards.
    # For a fixed W, we slide a window of size W across the period.
    # A window [S, S+W] is valid if we can assign distinct multiples to each divisor.
    
    # To speed up candidate generation, we can precompute nothing for now, 
    # as the window size W is small in the early iterations.
    
    # Maximum W to search. 
    # With 10 divisors and period 30030, the average gap between multiples is small.
    # A span of < 50 is extremely likely. We go up to 200 to be safe.
    max_w = 200
    
    for W in range(1, max_w + 1):
        # Check time
        if time.time() > deadline:
            break
            
        # Iterate over all possible window starts in the period
        # S is the minimum value in the window.
        # The window is [S, S+W].
        # We only need to check S in [0, period) because the problem is periodic.
        for S in range(period):
            if time.time() > deadline:
                break
                
            # For each divisor, find multiples in [S, S+W]
            candidates = []
            valid = True
            for i, d in enumerate(divisors):
                # Smallest multiple >= S
                k = (S + d - 1) // d
                v = k * d
                cands = []
                while v <= S + W:
                    cands.append(v)
                    v += d
                if not cands:
                    valid = False
                    break
                candidates.append(cands)
            
            if not valid:
                continue
            
            # Check if we can assign distinct values
            # Backtracking search
            # Order divisors by number of candidates (fewest first) to prune early
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            
            used = set()
            assignment = [0] * n
            
            def backtrack(idx):
                if idx == n:
                    return True
                i = order[idx]
                for v in candidates[i]:
                    if v not in used:
                        used.add(v)
                        assignment[i] = v
                        if backtrack(idx + 1):
                            return True
                        used.remove(v)
                        assignment[i] = 0
                return False
            
            if backtrack(0):
                # We found a valid assignment with span W
                # Construct the result list in the order of the original divisors
                result = [assignment[i] for i in range(n)]
                
                # Verify constraints
                # 1. Distinct
                if len(set(result)) != n:
                    continue # Should not happen due to backtracking logic
                # 2. Divisibility
                for i, d in enumerate(divisors):
                    if result[i] % d != 0:
                        continue # Should not happen
                # 3. Range
                if any(x < 1 or x > MAX_VAL for x in result):
                    continue # Should not happen for small W and S < period
                
                # Return as Python ints
                return [int(x) for x in result]
    
    # Fallback if no small W found (unlikely)
    # Greedy assignment for any valid range
    # This is just a safety net, should not be reached
    used = set()
    result = []
    for d in divisors:
        v = d
        while v in used or v > MAX_VAL:
            v += d
        result.append(int(v))
        used.add(v)
    return result

# EVOLVE-BLOCK-END
