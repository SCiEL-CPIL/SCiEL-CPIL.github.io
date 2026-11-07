"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Compute LCM of all divisors
    lcm_val = reduce(_lcm, divisors)
    # lcm_val should be 30030
    
    # Precompute: for each divisor d, which residues mod lcm_val are multiples of d
    # residue r is a multiple of d iff r % d == 0
    # For a window starting at s (0 <= s < lcm_val), the multiples of d in [s, s+R]
    # are values m such that m % d == 0 and s <= m <= s+R
    
    # For backtracking, for each divisor, given window [s, s+R], 
    # the candidates are: for k in range, m = k*d, check s <= m <= s+R
    # m = k*d, so k*d >= s and k*d <= s+R
    # k >= ceil(s/d) and k <= floor((s+R)/d)
    
    def get_candidates(d, s, R):
        """Get all multiples of d in [s, s+R]"""
        k_min = (s + d - 1) // d  # ceil(s/d)
        k_max = (s + R) // d       # floor((s+R)/d)
        if k_min > k_max:
            return []
        return [k * d for k in range(k_min, k_max + 1)]
    
    def feasible(s, R):
        """Check if we can assign distinct multiples to each divisor in window [s, s+R]"""
        # Get candidates for each divisor
        cands = []
        for d in divisors:
            c = get_candidates(d, s, R)
            if not c:
                return None
            cands.append(c)
        
        # Backtracking with pruning
        # Sort divisors by number of candidates (most constrained first)
        order = sorted(range(n), key=lambda i: len(cands[i]))
        used = set()
        assignment = [None] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            i = order[idx]
            for val in cands[i]:
                if val not in used:
                    used.add(val)
                    assignment[i] = val
                    if backtrack(idx + 1):
                        return True
                    used.remove(val)
                    assignment[i] = None
            return False
        
        if backtrack(0):
            return assignment
        return None
    
    # Try increasing window sizes R
    # Lower bound: at least 1 (since distinct)
    # Upper bound: something reasonable, say 10000
    
    best_result = None
    best_R = None
    
    # Try R from 1 upwards
    # But we can be smarter: try a range
    for R in range(1, 5000):
        if time.time() > deadline:
            break
        
        found = False
        for s in range(lcm_val):
            if time.time() > deadline:
                break
            result = feasible(s, R)
            if result is not None:
                # Verify distinctness and constraints
                vals = result
                if len(set(vals)) == n:
                    for i, d in enumerate(divisors):
                        if vals[i] % d != 0:
                            break
                    else:
                        if min(vals) > 0 and max(vals) <= 1000000:
                            best_result = vals
                            best_R = R
                            found = True
                            break
        if found:
            break
    
    if best_result is None:
        # Fallback: use the divisors themselves
        best_result = divisors[:]
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
