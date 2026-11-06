"""Closest strict underpacking of 1 by supplied unit fractions.
Return nonempty distinct zero-based indices into denominators [2,3,...,48]
whose reciprocal sum is STRICTLY below 1, minimizing the positive deficit."""
# EVOLVE-BLOCK-START
from fractions import Fraction
from time import time

_DEN = list(range(2, 49))          # denominators 2..48
_N = len(_DEN)                      # 47
_FRAC = [Fraction(1, d) for d in _DEN]
_ONE = Fraction(1, 1)

# suffix[i] = sum of _FRAC[i:]
_SUFFIX = [Fraction(0)] * (_N + 1)
for _i in range(_N - 1, -1, -1):
    _SUFFIX[_i] = _SUFFIX[_i + 1] + _FRAC[_i]


def _greedy():
    """Greedy maximal subset (include in order while sum stays < 1)."""
    path = []
    s = Fraction(0)
    for i in range(_N):
        ns = s + _FRAC[i]
        if ns < _ONE:
            s = ns
            path.append(i)
    return path, s


def _local_search(chosen, s, deadline):
    """Steepest-ascent on sum (descent on deficit) via add / 1-for-1 / 1-for-2."""
    chosen = set(chosen)
    f = _FRAC
    n = _N
    one = _ONE
    iters = 0
    while time() < deadline and iters < 400:
        iters += 1
        best = None  # (new_def, new_set, new_sum)
        # single adds
        for j in range(n):
            if j not in chosen:
                ns = s + f[j]
                if ns < one:
                    nd = one - ns
                    if best is None or nd < best[0]:
                        best = (nd, chosen | {j}, ns)
        if best is not None:
            _, chosen, s = best
            continue
        # 1-for-1 swaps
        for i in chosen:
            fi = f[i]
            for j in range(n):
                if j not in chosen:
                    ns = s - fi + f[j]
                    if ns < one and ns > s:
                        nd = one - ns
                        if best is None or nd < best[0]:
                            best = (nd, (chosen - {i}) | {j}, ns)
        if best is not None:
            _, chosen, s = best
            continue
        # 1-for-2 swaps
        for i in chosen:
            s1 = s - f[i]
            for j in range(n):
                if j not in chosen:
                    s2 = s1 + f[j]
                    if s2 < one:
                        for k in range(n):
                            if k not in chosen and k != j:
                                ns = s2 + f[k]
                                if ns < one and ns > s:
                                    nd = one - ns
                                    if best is None or nd < best[0]:
                                        best = (nd, (chosen - {i}) | {j, k}, ns)
        if best is not None:
            _, chosen, s = best
            continue
        break  # local optimum
    return chosen, s


def construct():
    t0 = time()
    bb_deadline = t0 + 54.0          # hard safety margin before the 60s kill

    # 1) initial bound from greedy
    g_path, g_sum = _greedy()
    best_def = _ONE - g_sum
    best_set = list(g_path)

    # 2) local search to tighten the bound (fast)
    ls_deadline = min(t0 + 2.5, bb_deadline - 2.0)
    ls_set, ls_sum = _local_search(g_path, g_sum, ls_deadline)
    ls_def = _ONE - ls_sum
    if 0 < ls_def < best_def:
        best_def = ls_def
        best_set = sorted(ls_set)

    # 3) exact branch-and-bound with strong admissible pruning
    frac = _FRAC
    suff = _SUFFIX
    n = _N
    one = _ONE
    path = []
    state = {'def': best_def, 'set': best_set, 'nodes': 0}
    dl = bb_deadline

    def dfs(i, cur):
        if time() > dl:
            return
        st = state
        st['nodes'] += 1
        # If even adding every remaining fraction stays below 1, adding all is
        # optimal for this branch -> capture and prune (no recursion needed).
        if cur + suff[i] < one:
            s = cur + suff[i]
            if s > 0:  # guarantees nonempty (only empty case: i==n and path empty)
                d = one - s
                if d < st['def']:
                    st['def'] = d
                    st['set'] = path + list(range(i, n))
            return
        # include branch (only if it stays strictly below 1)
        if cur + frac[i] < one:
            path.append(i)
            dfs(i + 1, cur + frac[i])
            path.pop()
        # exclude branch
        dfs(i + 1, cur)

    dfs(0, Fraction(0))

    result = state['set']
    if not result:  # defensive fallback (should never trigger)
        result = list(g_path)
    return result
# EVOLVE-BLOCK-END
