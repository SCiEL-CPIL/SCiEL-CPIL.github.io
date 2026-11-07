"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

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

    for R in range(1, 5000):
        if time.time() > deadline:
            break

        # Precompute candidate offsets for each (divisor i, residue r) for this R
        # cand_off[i][r] = tuple of offsets in [0, R-1] such that (L+r_offset) % d_i == 0
        # when L % d_i == r
        cand_off = []
        for i in range(n):
            d = divisors[i]
            row = []
            for r in range(d):
                fo = (d - r) % d
                if fo < R:
                    row.append(tuple(range(fo, R, d)))
                else:
                    row.append(())
            cand_off.append(row)

        for L in range(1, _LCM + 1):
            if time.time() > deadline:
                break

            # Build candidate values, quick-check all non-empty
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

            # MRV ordering
            order = sorted(range(n), key=lambda i: len(cands[i]))

            # Backtracking
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
                return [int(assigned[i]) for i in range(n)]

    return [int(d) for d in divisors]

# EVOLVE-BLOCK-END
