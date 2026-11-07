import time
import sys
import math
from itertools import permutations

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
    
    min_S = N - 1  # 9
    max_S = 500
    
    for S in range(min_S, max_S + 1):
        if time.time() > deadline:
            break
        
        # For each divisor, compute the set of valid window starts L (mod LCM)
        # where [L, L+S] contains at least one multiple of d
        # A multiple kd is in [L, L+S] iff L in [kd-S, kd]
        # So valid L mod LCM = union over k of [kd-S, kd] mod LCM
        
        # Represent valid L as a boolean array of length LCM
        # But LCM=30030, so we can use a bitset or boolean list
        
        valid_L = [True] * LCM
        
        for d in DIVISORS:
            # For this divisor, compute which L values (mod LCM) are valid
            d_valid = [False] * LCM
            # For each multiple kd of d, mark L in [max(1, kd-S), min(LCM, kd)] as valid
            # But we need to handle wrap-around since L is mod LCM
            # Actually, L ranges from 1 to LCM (window start), and we check if [L, L+S] has a multiple of d
            # Multiple kd is in [L, L+S] iff L <= kd <= L+S iff kd-S <= L <= kd
            # So for each k, L in [kd-S, kd] (clamped to [1, LCM])
            
            # Number of multiples of d in [1, LCM+S] to consider
            k_min = max(1, (1 + S) // d)  # smallest k such that kd >= 1
            k_max = (LCM + S) // d  # largest k such that kd <= LCM + S
            
            for k in range(k_min, k_max + 1):
                kd = k * d
                lo = max(1, kd - S)
                hi = min(LCM, kd)
                if lo <= hi:
                    for L in range(lo, hi + 1):
                        d_valid[L-1] = True
            
            # Intersect with valid_L
            for i in range(LCM):
                if not d_valid[i]:
                    valid_L[i] = False
        
        # Check if any L is valid
        # Also need L + S <= MAXV, so L <= MAXV - S
        max_L_allowed = min(LCM, MAXV - S)
        
        found = False
        for L in range(1, max_L_allowed + 1):
            if not valid_L[L-1]:
                continue
            
            # Now find distinct assignment using backtracking
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
            
            all_vals = set()
            for cands in candidates:
                all_vals.update(cands)
            if len(all_vals) < N:
                continue
            
            assignment = [None] * N
            used = set()
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
