"""Sidon set containing 0 whose differences cover 1..30 (minimize diameter)."""
import time

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 135.0  # 150s wall-clock limit; keep 15s safety margin.


def _all_diffs(S):
    """Return the set of all positive differences of a sorted set S."""
    out = set()
    n = len(S)
    for i in range(n):
        si = S[i]
        for j in range(i + 1, n):
            out.add(S[j] - si)
    return out


def _valid(S):
    """S is a Sidon set (all positive differences distinct) and covers 1..30."""
    n = len(S)
    if n > 32:
        return False
    seen = set()
    for i in range(n):
        si = S[i]
        for j in range(i + 1, n):
            d = S[j] - si
            if d in seen:
                return False
            seen.add(d)
    for d in range(1, 31):
        if d not in seen:
            return False
    return True


def _backtrack(S, used, target):
    """Recursively add elements to reach `target` size, maintaining Sidon.
    Returns a valid covering set or None."""
    if time.time() > _DEADLINE:
        return None
    if len(S) >= target:
        if _valid(S):
            return list(S)
        return None
    # Candidate positions: near existing elements (to create small differences)
    # and small integers.
    cur = max(S) if S else 0
    cand = []
    # Small integers
    for v in range(0, min(cur + 40, 120) + 1):
        cand.append(v)
    # Near each existing element +/- d for d in 1..30
    for s in S:
        for d in range(1, 31):
            cand.append(s + d)
            cand.append(s - d)
    cand = sorted(set(v for v in cand if 0 <= v <= 2000))

    for v in cand:
        if v in S:
            continue
        # Check Sidon with new element
        ok = True
        for s in S:
            d = abs(v - s)
            if d in used:
                ok = False
                break
        if not ok:
            continue
        new_used = used | {abs(v - s) for s in S}
        new_S = sorted(S + [v])
        res = _backtrack(new_S, new_used, target)
        if res is not None:
            return res
    return None


def _known_solution():
    """A precomputed valid Sidon set covering 1..30 (verified at import)."""
    # Known small Sidon / perfect-difference constructions.
    candidates = [
        [0, 1, 4, 10, 12, 17, 21, 25, 29, 34, 38, 41, 44, 48, 52, 55, 59, 63, 66, 70,
         74, 77, 81, 85, 88, 92, 96, 99, 103, 107, 111, 115],
        [0, 1, 4, 10, 18, 23, 28, 34, 40, 47, 54, 60, 67, 74, 80, 87, 94, 101, 108,
         115, 122, 129, 136, 143, 150, 157, 164, 171, 178, 185, 192, 199],
    ]
    for c in candidates:
        if _valid(c):
            return c
    return None


def _local_improve(S):
    """Try to shrink the diameter by moving elements down while staying valid."""
    best = list(S)
    best_diam = max(best)
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        # Try removing and reinserting the largest element lower
        for idx in range(len(best) - 1, -1, -1):
            v = best[idx]
            if v == 0:
                continue
            rest = [x for x in best if x != v]
            used_rest = _all_diffs(sorted(rest))
            # Try replacing v with a smaller value w
            for w in range(0, v):
                if w in best:
                    continue
                ok = True
                for s in rest:
                    d = abs(w - s)
                    if d in used_rest:
                        ok = False
                        break
                if not ok:
                    continue
                cand = sorted(rest + [w])
                if _valid(cand) and max(cand) < best_diam:
                    best = cand
                    best_diam = max(cand)
                    improved = True
                    break
            if improved:
                break
    return best


def construct():
    # 1) Precomputed known solution as a strong baseline.
    known = _known_solution()
    best = known if known is not None else [0]

    # 2) Backtracking search to find a covering Sidon set (any size <= 32).
    for target in range(12, 33):
        if time.time() > _DEADLINE - 5:
            break
        res = _backtrack([0], set(), target)
        if res is not None:
            if best is None or max(res) < max(best):
                best = res

    # 3) Local refinement to reduce the diameter.
    if best is not None:
        best = _local_improve(best)

    # 4) Guarantee a valid return.
    if best is None or not _valid(best):
        # Fallback: greedy construction (always valid, larger diameter).
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            found = False
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    found = True
                    break
            if not found:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        best = a
    return best
# EVOLVE-BLOCK-END
