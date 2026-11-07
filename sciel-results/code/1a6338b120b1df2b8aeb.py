import time
import sys
import math

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAXV = 1000000
N = len(DIVISORS)

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Calculate LCM of divisors
    LCM = 1
    for d in DIVISORS:
        LCM = _lcm(LCM, d)
        
    # LCM is 30030
    # We only need to scan L from 1 to LCM
    
    max_S = 200
    max_L = LCM
    
    for S in range(1, max_S + 1):
        if time.time() > deadline:
            break
        
        for L in range(1, max_L + 1):
            if time.time() > deadline:
                break
            
            if L + S > MAXV:
                break
            
            # Build candidates for each divisor
            candidates = []
            valid = True
            for d in DIVISORS:
                k = (L + d - 1) // d
                m = k * d
                cands = []
                while m <= L + S:
                    cands.append(m)
                    m += d
                if not cands:
                    valid = False
                    break
                candidates.append(cands)
            
            if not valid:
                continue
            
            # Backtracking to find distinct assignment
            assignment = [None] * N
            used = set()
            
            # Sort by number of candidates for better pruning
            indices = sorted(range(N), key=lambda i: len(candidates[i]))
            
            def backtrack(idx):
                if idx == len(indices):
                    return True
                i = indices[idx]
                for val in candidates[i]:
                    if val not in used:
                        used.add(val)
                        assignment[i] = val
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                        assignment[i] = None
                return False
            
            if backtrack(0):
                result = [int(assignment[i]) for i in range(N)]
                return result

    # Fallback: return divisors themselves
    return DIVISORS[:]
