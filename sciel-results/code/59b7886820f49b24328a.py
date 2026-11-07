import time
from math import gcd
from functools import reduce

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000

# Calculate LCM of divisors
def lcm(a, b):
    return a * b // gcd(a, b)

PERIOD = reduce(lcm, DIVISORS)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # We search for the minimum span W
    # The span W is max - min.
    # The window is [S, S+W].
    # We iterate W from 0 upwards.
    # For each W, we iterate S from 1 to PERIOD.
    # We check if there is a valid assignment.
    
    # Precompute divisors for faster access
    d = DIVISORS
    
    for W in range(0, 1000): # Upper bound for W, should be much smaller
        if time.time() > deadline:
            break
            
        # For this span W, check all possible starting positions S in one period
        # S is the minimum value in the window.
        # The window is [S, S+W].
        # Note: The values must be <= MAX_VAL.
        # Since we are looking for the *first* valid S in the period, 
        # and the pattern repeats every PERIOD, the first valid S 
        # will be the smallest S (geographically) that works, 
        # but we just need *any* valid assignment to return.
        # Actually, we just need *one* valid assignment for the minimal W.
        
        for S in range(1, PERIOD + 1):
            if time.time() > deadline:
                break
                
            # Check if S+W <= MAX_VAL is possible?
            # If S + W > MAX_VAL, we might still have a solution if we shift by -PERIOD?
            # No, the values must be positive.
            # The pattern of divisibility is periodic with PERIOD.
            # If [S, S+W] is valid, then [S+PERIOD, S+PERIOD+W] is also valid.
            # We want to return values <= 10^6.
            # The smallest valid S in [1, PERIOD] gives the smallest values.
            # So if S+W <= MAX_VAL, we are good.
            # If S+W > MAX_VAL, we might need to check if S - PERIOD >= 1?
            # But S is in [1, PERIOD], so S-PERIOD <= 0.
            # So if S+W > MAX_VAL, this specific S might be invalid due to upper bound.
            # However, since PERIOD is ~30k and MAX_VAL is 1M, 
            # and W is small, S+W will almost always be <= MAX_VAL for S in [1, PERIOD].
            # 30030 + 1000 < 1000000. So we are safe.
            
            if S + W > MAX_VAL:
                continue

            # Build domains for each divisor
            domains = []
            feasible = True
            for i in range(N):
                div = d[i]
                # Find multiples of div in [S, S+W]
                # First multiple >= S:
                # k = ceil(S / div)
                k = (S + div - 1) // div
                val = k * div
                dom = []
                while val <= S + W:
                    dom.append(val)
                    val += div
                if not dom:
                    feasible = False
                    break
                domains.append(dom)
            
            if not feasible:
                continue

            # Backtracking to find distinct representatives
            # Sort variables by domain size (MRV)
            # We need to keep track of original indices to return in order
            indices = list(range(N))
            indices.sort(key=lambda i: len(domains[i]))
            
            used = set()
            assignment = [0] * N
            
            def backtrack(idx):
                if idx == N:
                    return True
                i = indices[idx]
                for val in domains[i]:
                    if val not in used:
                        used.add(val)
                        assignment[i] = val
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                return False

            if backtrack(0):
                # Found a valid assignment for this W
                # Return in the order of DIVISORS
                result = [assignment[i] for i in range(N)]
                # Ensure types are int
                return [int(x) for x in result]
    
    # Fallback if no solution found within limits (should not happen)
    # Just return the divisors themselves as a valid (though not optimal) set?
    # The divisors are distinct multiples of themselves.
    # Span would be max(d) - min(d) = 39 - 6 = 33.
    # This is a valid solution.
    return [int(x) for x in DIVISORS]
