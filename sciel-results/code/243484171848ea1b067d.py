import time
import math
from functools import reduce

def construct():
    """
    Finds 10 distinct positive integers <= 1000000, each divisible by its corresponding divisor in [6,10,14,15,21,22,26,33,35,39],
    such that max(reps) - min(reps) is minimized.
    """
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    deadline = time.time() + 85  # Safety margin under 90s limit
    
    lcm_val = reduce(lambda a, b: a * b // math.gcd(a, b), divisors)
    
    best_span = float('inf')
    best_result = None
    
    def check_span(S):
        """Try to find a valid assignment with span <= S. Returns reps or None."""
        # Generate candidate L values: only check L where feasibility can change
        # For each divisor d, critical L values are k*d and k*d - S for k >= 1
        # We only need L in [1, lcm_val]
        critical_L = set()
        for d in divisors:
            k_min = 1
            k_max = lcm_val // d
            for k in range(k_min, k_max + 1):
                val = k * d
                if 1 <= val <= lcm_val:
                    critical_L.add(val)
                val2 = val - S
                if 1 <= val2 <= lcm_val:
                    critical_L.add(val2)
        
        # Also add L=1 and L=lcm_val
        critical_L.add(1)
        critical_L.add(lcm_val)
        
        for L in sorted(critical_L):
            if time.time() > deadline:
                return None
            upper = L + S
            if upper > max_val:
                continue
            
            candidates = []
            feasible = True
            for d in divisors:
                first = ((L + d - 1) // d) * d
                if first > upper:
                    feasible = False
                    break
                opts = []
                m = first
                while m <= upper:
                    opts.append(m)
                    m += d
                candidates.append(opts)
            
            if not feasible:
                continue
            
            # Try to assign distinct values
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            used = set()
            reps = [0] * n
            
            def bt(idx):
                if idx == n:
                    return True
                for m in candidates[order[idx]]:
                    if m not in used:
                        used.add(m)
                        reps[order[idx]] = m
                        if bt(idx + 1):
                            return True
                        used.remove(m)
                return False
            
            if bt(0):
                return reps[:]
        
        return None
    
    # Try small spans first
    for S in range(1, 60):
        if time.time() > deadline:
            break
        res = check_span(S)
        if res is not None:
            best_span = S
            best_result = res
            break
    
    if best_result is None:
        for S in range(60, 200):
            if time.time() > deadline:
                break
            res = check_span(S)
            if res is not None:
                best_span = S
                best_result = res
                break
    
    if best_result is None:
        best_result = divisors[:]
    
    return [int(x) for x in best_result]
