"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

_DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_N = len(_DIVISORS)
_MAX_VAL = 1000000

def _lcm(a, b):
    return a * b // math.gcd(a, b)

_LCM = 1
for d in _DIVISORS:
    _LCM = _lcm(_LCM, d)

def construct():
    deadline = time.time() + 85
    divisors = _DIVISORS
    n = len(divisors)
    
    # Precompute for each divisor, for each possible residue L % d, the offsets
    # in [0, R-1] where (L + offset) % d == 0
    # We'll compute this on the fly for each R since it depends on R
    
    best_result = None
    
    for R in range(1, 200):
        if time.time() > deadline - 5:
            break
        
        # Precompute candidate offsets for each (divisor i, residue r) for this R
        cand_off = []
        for i in range(n):
            d = divisors[i]
            row = []
            for r in range(d):
                # We need (L + o) % d == 0 where L % d == r
                # So o % d == (d - r) % d
                fo = (d - r) % d
                if fo < R:
                    row.append(tuple(range(fo, R, d)))
                else:
                    row.append(())
            cand_off.append(row)
        
        for L in range(1, _LCM + 1):
            if time.time() > deadline - 5:
                break
            
            # Build candidate values for each divisor
            cands = [None] * n
            ok = True
            for i in range(n):
                d = divisors[i]
                r = L % d
                offs = cand_off[i][r]
                if not offs:
                    ok = False
                    break
                cands[i] = tuple(L + o for o in offs)
            if not ok:
                continue
            
            # Quick check: all must have at least one candidate
            # MRV ordering
            order = sorted(range(n), key=lambda i: len(cands[i]))
            
            # Backtracking with MRV
            used = set()
            assigned = [0] * n
            
            def bt(idx):
                if idx == n:
                    return max(assigned) <= _MAX_VAL
                i = order[idx]
                for val in cands[i]:
                    if val not in used:
                        used.add(val)
                        assigned[i] = val
                        if bt(idx + 1):
                            return True
                        used.remove(val)
                return False
            
            if bt(0):
                result = [int(assigned[i]) for i in range(n)]
                # Verify
                if all(result[i] % divisors[i] == 0 for i in range(n)):
                    if all(isinstance(x, int) for x in result):
                        if len(set(result)) == n:
                            if max(result) <= _MAX_VAL and min(result) >= 1:
                                return result
    
    # Fallback: return the divisors themselves (not optimal but valid)
    return [int(d) for d in divisors]

# EVOLVE-BLOCK-END
