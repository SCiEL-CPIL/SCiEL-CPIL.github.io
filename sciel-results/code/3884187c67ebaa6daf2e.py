"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import lru_cache

def _lcm(a, b):
    return a // gcd(a, b) * b

def _compute_lcm(lst):
    r = 1
    for x in lst:
        r = _lcm(r, x)
    return r

_DIVS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_LCM = _compute_lcm(_DIVS)  # 30030
_N = len(_DIVS)
_MAX_VAL = 1000000

def construct():
    t_start = time.time()
    deadline = t_start + 85.0  # safety margin before 90s kill

    # For a window [s, s+W], divisor d has a multiple in it at position:
    #   m_d = d * ceil(s/d)
    # This is valid iff m_d <= s + W, i.e., d*ceil(s/d) - s <= W
    # Let r = s mod d. Then ceil(s/d) = (s + d - r) // d if r > 0, else s//d
    # m_d - s = d*ceil(s/d) - s
    # If r == 0: m_d - s = 0
    # If r > 0: m_d - s = d - r
    # So the condition is: (d - (s mod d)) if (s mod d) > 0 else 0 <= W
    # Equivalently: s mod d >= d - W (when W < d), or always true (when W >= d)

    # For W < min(_DIVS) = 6, every divisor requires s mod d >= d - W
    # For larger W, some divisors are always satisfied.

    # Strategy: iterate W from 1 upward. For each W, find if there exists s in [0, LCM)
    # such that all constraints are met and all resulting multiples are distinct.

    best_result = None
    best_W = None

    for W in range(1, 200):
        if time.time() > deadline:
            break

        # Precompute for each divisor d: the set of residues r = s mod d that are valid
        # Valid iff (d - r) <= W when r > 0, or 0 <= W when r == 0
        # i.e., r >= d - W (and r >= 0), with r=0 always valid
        # So valid residues: {0} ∪ {d-W, d-W+1, ..., d-1} if W < d, else all residues

        # For each s, compute the multiple: m_d = d * ceil(s/d)
        # Two divisors d_i, d_j collide iff d_i * ceil(s/d_i) == d_j * ceil(s/d_j)

        # Efficient approach: for each s in [0, LCM), check all constraints.
        # But LCM = 30030, and for each W we scan all s. Total work ~ 200 * 30030 * 10 = 60M ops.
        # This might be too slow in Python. Let's optimize.

        # Optimization: precompute for each s the value s mod d for all d, and the
        # resulting multiple. Then for each W, just check the max offset and distinctness.

        # Actually, let's precompute for each s in [0, LCM):
        #   offsets[s] = list of (d - s%d) if s%d > 0 else 0, for each d
        #   multiples[s] = list of d * ceil(s/d), for each d
        # Then for a given W, s is feasible iff:
        #   max(offsets[s]) <= W
        #   all elements of multiples[s] are distinct

        # Precompute once:
        # offsets[s][i] = (d_i - s % d_i) if s % d_i != 0 else 0
        # multiples[s][i] = d_i * ((s + d_i - 1) // d_i)  [this is d_i * ceil(s/d_i)]

        # But storing 30030 * 10 = 300K values is fine.

        # Let's precompute the maximum offset for each s and check distinctness of multiples.
        # max_offset[s] = max over i of (d_i - s%d_i if s%d_i != 0 else 0)
        # For a given W, s is a candidate iff max_offset[s] <= W.
        # Then we need to check if multiples[s] are all distinct.

        # Precompute max_offset and a hash/fingerprint of the multiples set for distinctness.
        # Actually, just check distinctness on the fly for candidates.

        # Precompute max_offset for all s:
        max_offset = [0] * _LCM
        for s in range(_LCM):
            mo = 0
            for d in _DIVS:
                r = s % d
                off = 0 if r == 0 else d - r
                if off > mo:
                    mo = off
            max_offset[s] = mo

        # Now for each W, find s with max_offset[s] <= W, then check distinctness.
        # To speed up, for each W, we can iterate s and check max_offset[s] <= W.
        # But that's still 30030 per W. Let's precompute which s are candidates for each W.

        # Alternative: sort s by max_offset[s], then for each W, the candidates are
        # those with max_offset <= W. But we need to check distinctness for each.

        # Let's just iterate. 200 * 30030 = 6M iterations, each with O(1) check for
        # max_offset and O(10) for distinctness. Should be fine.

        found = False
        for s in range(_LCM):
            if time.time() > deadline:
                break

            if max_offset[s] > W:
                continue

            # Compute multiples and check distinctness
            vals = []
            valid = True
            for d in _DIVS:
                r = s % d
                if r == 0:
                    m = s
                else:
                    m = s + (d - r)
                vals.append(m)

            if len(set(vals)) != _N:
                continue

            # Check all <= MAX_VAL
            if max(vals) > _MAX_VAL:
                continue

            # Valid solution found
            found = True
            best_W = W
            best_result = vals
            break

        if found:
            break

    if best_result is not None:
        return [int(x) for x in best_result]

    # Fallback: return the divisors themselves
    return [int(d) for d in _DIVS]
# EVOLVE-BLOCK-END
