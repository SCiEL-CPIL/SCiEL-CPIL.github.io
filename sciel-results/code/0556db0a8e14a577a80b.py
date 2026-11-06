"""Sidon set in [1,284] containing {1,2,5,11}; maximize size.

Pairwise sums a+b (a<=b) must all be distinct. The constraint is equivalent
to a Golomb-ruler on the transformed values v = 2*x - 1 (all odd), because
2*(a+b) = (2a-1)+(2b+1) is unique iff all pairwise sums of v are unique.
"""
import time
import random

_START = [1, 2, 5, 11]
_U = 284
_N = 568
_RESERVED = {1, 2, 5, 11}
_DEADLINE = 135.0  # wall-clock safety margin under the 150s kill


def _to_v(x):
    return 2 * x - 1


def _from_v(v):
    return (v + 1) // 2


def _build(v):
    """Return (sorted x-list, sum-set) for a list of odd v-values."""
    s = set()
    for i in range(len(v)):
        vi = v[i]
        s.add(2 * vi)
        for j in range(i + 1, len(v)):
            s.add(vi + v[j])
    xs = sorted(_from_v(t) for t in v)
    return xs, s


def _valid(xs):
    """Verify the Sidon condition directly on x-values."""
    n = len(xs)
    s = set()
    for i in range(n):
        a = xs[i]
        s.add(2 * a)
        for j in range(i + 1, n):
            s.add(a + xs[j])
    return len(s) == n * (n + 1) // 2


def _greedy(v, rng):
    """Greedily add a random valid odd value until stuck."""
    s = set()
    for i in range(len(v)):
        vi = v[i]
        s.add(2 * vi)
        for j in range(i + 1, len(v)):
            s.add(vi + v[j])
    cur = list(v)
    cand = [x for x in range(1, _N + 1, 2) if x not in _RESERVED]
    while True:
        rng.shuffle(cand)
        added = False
        for c in cand:
            if c in s:
                continue
            ok = True
            for t in cur:
                if c + t in s:
                    ok = False
                    break
            if ok:
                cur.append(c)
                s.add(2 * c)
                for t in cur:
                    if t != c:
                        s.add(c + t)
                cand.remove(c)
                added = True
                break
        if not added:
            break
    return cur


def _local_search(cur, s, rng, deadline):
    """Swap-based local search on odd v-values."""
    n = len(cur)
    used = set(cur)
    best = n
    while time.time() < deadline:
        cur2 = list(cur)
        s2 = set(s)
        improved = False
        rng.shuffle(cur2)
        for i in range(n):
            a = cur2[i]
            if a not in s2:
                continue
            rem = [t for t in cur2 if t != a]
            for c in range(1, _N + 1, 2):
                if c in used or c in s2:
                    continue
                ok = True
                for t in rem:
                    if c + t in s2:
                        ok = False
                        break
                if not ok:
                    continue
                # tentatively swap
                for t in rem:
                    s2.remove(a + t)
                s2.remove(2 * a)
                for t in rem:
                    s2.add(c + t)
                s2.add(2 * c)
                if len(s2) == n * (n + 1) // 2:
                    cur2[i] = c
                    used.remove(a)
                    used.add(c)
                    s2 = set(s2)
                    s = s2
                    improved = True
                    break
                else:
                    # undo
                    for t in rem:
                        s2.remove(c + t)
                    s2.remove(2 * c)
                    for t in rem:
                        s2.add(a + t)
                    s2.add(2 * a)
            if improved:
                break
        cur = cur2
        if len(cur) > best:
            best = len(cur)
        if not improved:
            # small random perturbation to escape local optimum
            if len(cur) >= 4:
                i = rng.randrange(n)
                a = cur[i]
                for c in range(1, _N + 1, 2):
                    if c in used:
                        continue
                    rem = [t for t in cur if t != a]
                    ok = True
                    for t in rem:
                        if c + t in s:
                            ok = False
                            break
                    if not ok:
                        continue
                    for t in rem:
                        s.remove(a + t)
                    s.remove(2 * a)
                    for t in rem:
                        s.add(c + t)
                    s.add(2 * c)
                    cur[i] = c
                    used.remove(a)
                    used.add(c)
                    break
    return cur, s


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE

    base_v = [_to_v(x) for x in _START]

    # 1) deterministic greedy baseline
    base = _greedy(base_v, random.Random(0))
    xs, s = _build(base)
    best_xs, best_s, best_n = xs, s, len(xs)
    if not _valid(best_xs):
        best_xs, best_s, best_n = list(_START), set(), 4

    # 2) randomized greedy restarts + local search refinement
    rng = random.Random(12345)
    while time.time() < deadline:
        cand = _greedy(base_v, rng)
        xs, s = _build(cand)
        if len(xs) > best_n:
            best_xs, best_s, best_n = xs, s, len(xs)
        # refine the current candidate with local search
        cand2, s2 = _local_search(cand, s, rng, deadline)
        xs2, s2b = _build(cand2)
        if len(xs2) > best_n:
            best_xs, best_s, best_n = xs2, s2b, len(xs2)

    # final safety check
    if not _valid(best_xs):
        # fall back to a known-valid greedy result
        xs, s = _build(_greedy(base_v, random.Random(7)))
        if _valid(xs):
            best_xs = xs
        else:
            best_xs = list(_START)

    return best_xs


# EVOLVE-BLOCK-END
