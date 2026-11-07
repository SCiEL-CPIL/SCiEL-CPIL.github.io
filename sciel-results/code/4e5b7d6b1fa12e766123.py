"""Pack distinct multiples of a fixed integer set - minimize the enclosing span."""
import time

# EVOLVE-BLOCK-START
_DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_MAXV = 1000000
_N = len(_DIVISORS)
_PERIOD = 30030
_DEADLINE = time.time() + 85
_BEST = list(_DIVISORS)
_BEST_SPAN = max(_BEST) - min(_BEST)


def _check_window(L, S):
    U = L + S
    cands = []
    for d in _DIVISORS:
        k = (L + d - 1) // d
        multiples = []
        m = k * d
        while m <= U:
            multiples.append(m)
            m += d
        cands.append(multiples)
    for c in cands:
        if not c:
            return None
    order = sorted(range(_N), key=lambda i: len(cands[i]))
    used = set()
    reps = [0] * _N
    def bt(pos):
        if pos == _N:
            return True
        i = order[pos]
        for val in cands[i]:
            if val not in used:
                used.add(val)
                reps[i] = val
                if bt(pos + 1):
                    return True
                used.remove(val)
        return False
    if bt(0):
        return reps
    return None


def _get_critical_Ls(S):
    critical = set()
    max_L = min(PERIOD, _MAXV - S)
    if max_L < 1:
        return []
    for d in _DIVISORS:
        min_m = 1 + S
        max_m = min(_MAXV, max_L + S)
        if min_m > max_m:
            continue
        k_start = (min_m + d - 1) // d
        k_end = max_m // d
        for k in range(k_start, k_end + 1):
            m = k * d
            L1 = m - S
            if 1 <= L1 <= max_L:
                critical.add(L1)
            L2 = m + 1
            if 1 <= L2 <= max_L:
                critical.add(L2)
    return sorted(critical)


def construct():
    global _BEST, _BEST_SPAN
    start = time.time()
    for S in range(1, _BEST_SPAN):
        if time.time() > _DEADLINE:
            break
        critical_Ls = _get_critical_Ls(S)
        found = False
        for L in critical_Ls:
            if time.time() > _DEADLINE:
                break
            res = _check_window(L, S)
            if res is not None:
                if all(1 <= x <= _MAXV for x in res) and len(set(res)) == _N:
                    if S < _BEST_SPAN:
                        _BEST = list(res)
                        _BEST_SPAN = S
                found = True
                break
        if found:
            break
    out = [int(x) for x in _BEST]
    assert len(out) == _N
    assert all(isinstance(x, int) and not isinstance(x, bool) for x in out)
    assert len(set(out)) == _N
    assert all(1 <= x <= _MAXV for x in out)
    assert all(out[i] % _DIVISORS[i] == 0 for i in range(_N))
    return out
# EVOLVE-BLOCK-END
