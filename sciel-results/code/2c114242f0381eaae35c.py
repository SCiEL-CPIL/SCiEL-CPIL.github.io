"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

_DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_LCM = 30030
_MAX_VAL = 1000000
_N = 10

# Precompute residues for each divisor
_residues = []
for d in _DIVISORS:
    res = list(range(0, _LCM, d))
    _residues.append(res)

# Sort indices by scarcity (fewest residues first) for better backtracking
_indices = sorted(range(_N), key=lambda i: len(_residues[i]))
_inv_indices = [0] * _N
for new_idx, orig_idx in enumerate(_indices):
    _inv_indices[orig_idx] = new_idx


def _check_window(W):
    """
    Check if there is a window of length W that allows distinct assignment.
    Returns (offset, assignment_list) where assignment_list is in original divisor order.
    """
    for S in range(_LCM):
        used = set()
        assignment = [None] * _N

        def backtrack(idx):
            if idx == _N:
                return True

            d_idx = _indices[idx]
            res_list = _residues[d_idx]

            for p in res_list:
                if (p - S) % _LCM <= W:
                    if p not in used:
                        used.add(p)
                        assignment[d_idx] = p
                        if backtrack(idx + 1):
                            return True
                        used.remove(p)
                        assignment[d_idx] = None
            return False

        if backtrack(0):
            return S, assignment

    return None


def construct():
    start_time = time.time()
    deadline = start_time + 85

    best_W = None
    best_S = None
    best_assignment = None

    lo = 1
    hi = _LCM

    # Binary search on W
    while lo < hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        res = _check_window(mid)
        if res is None:
            if time.time() > deadline:
                break
            lo = mid + 1
        else:
            S, assignment = res
            best_W = mid
            best_S = S
            best_assignment = assignment
            hi = mid

    if best_W is None:
        return list(_DIVISORS)

    # Construct actual integers
    min_p = min(best_assignment)
    max_p = max(best_assignment)

    # Find smallest S_base (multiple of LCM) such that S_base + min_p >= 1
    # and S_base + max_p <= 1000000
    min_needed = 1 - min_p
    max_allowed = _MAX_VAL - max_p

    if min_needed <= 0:
        S_base = 0
    else:
        S_base = ((min_needed + _LCM - 1) // _LCM) * _LCM

    if S_base > max_allowed:
        return list(_DIVISORS)

    result = [S_base + p for p in best_assignment]

    # Verify
    for i, val in enumerate(result):
        if val <= 0 or val > _MAX_VAL:
            return list(_DIVISORS)
        if val % _DIVISORS[i] != 0:
            return list(_DIVISORS)
    if len(set(result)) != _N:
        return list(_DIVISORS)

    return result

# EVOLVE-BLOCK-END
