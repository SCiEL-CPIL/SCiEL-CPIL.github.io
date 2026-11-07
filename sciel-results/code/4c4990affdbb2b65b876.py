import time
import math
import numpy as np

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    start_time = time.time()
    deadline = start_time + 55.0
    
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    lcm_val = 1
    for d in divisors:
        lcm_val = lcm(lcm_val, d)
    
    # Known good solution with range 19
    known_good = [24, 40, 28, 30, 21, 22, 26, 33, 35, 39]
    best_range = max(known_good) - min(known_good)
    best_vals = known_good[:]
    
    # Precompute multiples for each divisor
    max_W = 25
    multiples_by_divisor = []
    for d in divisors:
        m = 0
        mults = []
        while m <= lcm_val + max_W:
            mults.append(m)
            m += d
        multiples_by_divisor.append(np.array(mults, dtype=np.int32))
    
    # Search for better solutions
    for W in range(9, max_W + 1):
        if time.time() > deadline:
            break
        if W >= best_range:
            break
        
        for A in range(0, lcm_val):
            if time.time() > deadline:
                break
            
            candidates = []
            ok = True
            for d_idx, d in enumerate(divisors):
                mults = multiples_by_divisor[d_idx]
                left = np.searchsorted(mults, A, side='left')
                right = np.searchsorted(mults, A + W, side='right')
                if left >= right:
                    ok = False
                    break
                candidates.append(mults[left:right])
            
            if not ok:
                continue
            
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            
            if len(candidates[order[0]]) > 3:
                continue
            
            used = set()
            assignment = [None] * n
            
            def bt(idx):
                if idx == n:
                    return True
                i = order[idx]
                for v in candidates[i]:
                    vi = int(v)
                    if vi not in used:
                        used.add(vi)
                        assignment[i] = vi
                        if bt(idx + 1):
                            return True
                        used.discard(vi)
                        assignment[i] = None
                return False
            
            if bt(0):
                vals = assignment[:]
                r = max(vals) - min(vals)
                if r < best_range:
                    best_range = r
                    best_vals = vals[:]
        
        if best_range <= W:
            break
    
    return [int(v) for v in best_vals]
