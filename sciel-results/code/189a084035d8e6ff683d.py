"""Pre-computed minimal antichain representations for targets 1..128."""

import time
from itertools import combinations

_ANSWERS = None

def _generate_smooth_numbers(limit):
    smooth = []
    for x in range(1, limit + 1):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    return smooth

def _build_divisibility_masks(smooth, n):
    """For each smooth number (as index), a bitmask of indices that divide it."""
    m = len(smooth)
    masks = [0] * m
    for i in range(m):
        for j in range(m):
            if i != j and smooth[i] % smooth[j] == 0:
                masks[i] |= (1 << j)
    return masks

def _find_min_antichain(n, smooth, masks, index_of):
    candidates = [x for x in smooth if x <= n]
    cand_indices = [index_of[x] for x in candidates]

    # 1 term
    if n in index_of:
        return [n]

    # 2 terms: for each a, check if n-a is a smooth candidate and antichain
    cand_set = set(candidates)
    for a in candidates:
        b = n - a
        if b in cand_set and b != a:
            ia = index_of[a]
            ib = index_of[b]
            # antichain: a does not divide b and b does not divide a
            if (masks[ia] & (1 << ib)) == 0 and (masks[ib] & (1 << ia)) == 0:
                return [a, b]

    # 3 terms: for each pair (a,b), check if n-a-b is a smooth candidate and antichain with both
    # To keep it fast, iterate a and b from candidates, but prune by sum
    for a in candidates:
        if a >= n:
            break
        for b in candidates:
            if b == a:
                continue
            s2 = a + b
            if s2 >= n:
                continue
            c = n - s2
            if c in cand_set:
                ia = index_of[a]
                ib = index_of[b]
                ic = index_of[c]
                # check antichain: no term divides another
                if (masks[ia] & (1 << ib)) != 0:
                    continue
                if (masks[ia] & (1 << ic)) != 0:
                    continue
                if (masks[ib] & (1 << ic)) != 0:
                    continue
                return [a, b, c]

    # 4 terms: combinations fallback
    for a, b, c, d in combinations(candidates, 4):
        if a + b + c + d != n:
            continue
        ia, ib, ic, id_ = index_of[a], index_of[b], index_of[c], index_of[d]
        # antichain check
        ok = True
        for x, y in ((ia, ib), (ia, ic), (ia, id_), (ib, ic), (ib, id_), (ic, id_)):
            if (masks[x] & (1 << y)) != 0 or (masks[y] & (1 << x)) != 0:
                ok = False
                break
        if ok:
            return [a, b, c, d]

    return None

def _precompute():
    global _ANSWERS
    if _ANSWERS is not None:
        return _ANSWERS

    start_time = time.time()
    deadline = start_time + 70.0  # safety margin under 90s

    smooth = _generate_smooth_numbers(128)
    index_of = {x: i for i, x in enumerate(smooth)}
    masks = _build_divisibility_masks(smooth, 128)

    answers = []
    for n in range(1, 129):
        if time.time() > deadline:
            # Fallback: use a known valid representation if possible
            # For safety, if we run out of time, try a simple greedy or reuse
            # But we expect to finish well within time.
            ans = _find_min_antichain(n, smooth, masks, index_of)
            if ans is None:
                ans = [n] if n in index_of else [1, n-1] if (n-1) in index_of else [n]
        else:
            ans = _find_min_antichain(n, smooth, masks, index_of)
            if ans is None:
                # Should not happen; fallback to a valid smooth sum if possible
                ans = [n] if n in index_of else [1, n-1] if (n-1) in index_of else [n]
        answers.append(ans)

    _ANSWERS = answers
    return _ANSWERS

def construct():
    return _precompute()
