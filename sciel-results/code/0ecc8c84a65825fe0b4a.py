import random
import time
from itertools import combinations

DEADLINE = 120.0
N = 30
NEED = 31          # 0 plus 30 difference-carrying elements
RANGE = 10**6


def _now():
    return time.time()


def _is_sidon(s):
    # all positive differences distinct
    d = []
    for i in range(len(s)):
        si = s[i]
        for j in range(i + 1, len(s)):
            d.append(s[j] - si)
    return len(set(d)) == len(d)


def _diffs(s):
    return [s[j] - s[i] for i in range(len(s)) for j in range(i + 1, len(s))]


def _missing(s, targets):
    have = set(_diffs(s))
    return [t for t in targets if t not in have]


def _has_diff(s, d):
    st = set(s)
    return any((x + d) in st for x in s)


def _add_safe(s, x, targets):
    t = sorted(s + [x])
    if _is_sidon(t) and _missing(t, targets):
        return t
    return None


def _greedy(s, targets, max_val, rng):
    s = list(s)
    for d in targets:
        if _has_diff(s, d):
            continue
        cands = []
        for v in s:
            for cand in (v + d, v - d):
                if 0 <= cand <= max_val and cand not in s:
                    cands.append(cand)
        rng.shuffle(cands)
        placed = False
        for cand in cands:
            t = _add_safe(s, cand, targets)
            if t is not None:
                s = t
                placed = True
                break
        if not placed:
            lo = max(0, min(s) - d)
            hi = min(max_val, max(s) + d)
            for cand in range(lo, hi + 1):
                if cand in s:
                    continue
                t = _add_safe(s, cand, targets)
                if t is not None:
                    s = t
                    placed = True
                    break
        if not placed:
            return None
    return s


def _build_half(targets, max_val, rng):
    best = None
    for _ in range(6):
        s = [0]
        r = _greedy(s, list(targets), max_val, rng)
        if r is not None:
            if best is None or max(r) < max(best):
                best = r
    return best


def _merge(a, b, targets):
    # a already has 0; shift b by t so b has no 0 and no overlap with a
    used = set(a)
    best = None
    for t in range(1, 400):
        sb = [x + t for x in b]
        if any(x in used for x in sb):
            continue
        merged = sorted(a + sb)
        if _is_sidon(merged) and _missing(merged, targets) == [] and len(merged) <= 32:
            if best is None or max(merged) < max(best):
                best = merged
    return best


def _local_search(s, targets, budget, rng):
    s = list(s)
    best = list(s)
    t0 = _now()
    improved = True
    while improved and _now() < budget:
        improved = False
        # try removing each non-zero element
        for i in range(len(s)):
            if s[i] == 0:
                continue
            cand = s[:i] + s[i + 1:]
            if len(cand) < NEED:
                continue
            if _is_sidon(cand) and _missing(cand, targets) == []:
                s = cand
                improved = True
                break
        if improved:
            continue
        # try moving each non-zero element to a smaller value
        for i in range(len(s)):
            if s[i] == 0:
                continue
            old = s[i]
            for nv in range(0, old):
                if nv in s:
                    continue
                cand = s[:i] + [nv] + s[i + 1:]
                if _is_sidon(cand) and _missing(cand, targets) == []:
                    s = cand
                    improved = True
                    break
            if improved:
                break
    if max(s) < max(best):
        best = s
    return best


def construct():
    start = _now()
    hard = start + DEADLINE
    rng = random.Random(12345)
    targets = list(range(1, N + 1))

    # --- Phase 1: meet-in-the-middle construction ---
    half1 = targets[:15]
    half2 = targets[15:]
    best = None

    for m1 in (120, 160, 200, 260, 340):
        if _now() > hard - 30:
            break
        a = _build_half(half1, m1, rng)
        if a is None:
            continue
        for m2 in (m1, m1 + 40, m1 + 90, m1 + 150):
            if _now() > hard - 30:
                break
            b = _build_half(half2, m2, rng)
            if b is None:
                continue
            merged = _merge(a, b, targets)
            if merged is not None:
                if best is None or max(merged) < max(best):
                    best = merged

    # --- Phase 2: global greedy constructions as backups ---
    for _ in range(40):
        if _now() > hard - 30:
            break
        r = _greedy([0], list(targets), 3000, rng)
        if r is not None:
            if best is None or max(r) < max(best):
                best = r

    if best is None:
        # last-resort deterministic greedy
        r = _greedy([0], list(targets), RANGE, rng)
        if r is not None:
            best = r
        else:
            # extremely unlikely; return a trivially valid fallback
            return list(range(0, NEED))

    # --- Phase 3: local search to shave diameter ---
    if _now() < hard - 5:
        best = _local_search(best, targets, hard - 5, rng)

    return sorted(best)
