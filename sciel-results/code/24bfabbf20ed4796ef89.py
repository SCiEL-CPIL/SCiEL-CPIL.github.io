"""Fresh design: randomized greedy construct + element-removal minimization."""
import time
import random

_DEADLINE = 150.0
_T0 = time.time()
_TARGETS = tuple(range(1, 31))


def _diffs(s):
    n = len(s)
    out = [0] * (n * (n - 1) // 2)
    k = 0
    for i in range(n):
        si = s[i]
        for j in range(i + 1, n):
            out[k] = s[j] - si
            k += 1
    return out


def _is_sidon(s):
    return len(set(_diffs(s))) == len(_diffs(s))


def _covers(s):
    return set(_diffs(s)) >= set(_TARGETS)


def _valid(s):
    return len(s) <= 32 and 0 in s and _is_sidon(s) and _covers(s)


def _best(a):
    return max(a)


def _greedy(seed, rng):
    """Randomized greedy build; returns list or None."""
    a = [0]
    present = {0}
    diffset = set()
    for d in _TARGETS:
        if d in diffset:
            continue
        # candidate values: place a new element to create difference d
        cands = set()
        for v in a:
            for x in (v + d, v - d):
                if x >= 0 and x not in present:
                    cands.add(x)
        # order candidates by how tight they keep the diameter
        order = sorted(cands, key=lambda x: (max(a + [x]), x))
        placed = False
        for x in order:
            # check collision
            ok = True
            for v in a:
                if x - v in diffset or v - x in diffset:
                    ok = False
                    break
            if not ok:
                continue
            # commit
            a.append(x)
            present.add(x)
            for v in a:
                if v != x:
                    diffset.add(abs(x - v))
            placed = True
            break
        if not placed:
            # fallback: search a fresh value far out
            y = 2 * max(a) + 3 * d + 1
            while True:
                ok = True
                for v in a:
                    if y - v in diffset or (v - y) in diffset or (y - v) == 0:
                        ok = False
                        break
                if ok:
                    a.append(y)
                    present.add(y)
                    for v in a:
                        if v != y:
                            diffset.add(abs(y - v))
                    break
                y += 1
    return a


def _remove_pass(a, rng):
    """Try to remove redundant elements while staying valid; minimize max."""
    a = list(a)
    improved = True
    while improved:
        improved = False
        # candidate elements to remove (never 0)
        cands = [x for x in a if x != 0]
        rng.shuffle(cands)
        for x in cands:
            trial = [v for v in a if v != x]
            if len(trial) < 1:
                continue
            if not _is_sidon(trial):
                continue
            if not _covers(trial):
                continue
            a = trial
            improved = True
            break
    return a


def _local_shrink(a, rng, budget):
    """Try to lower the diameter by repositioning the largest element."""
    a = list(a)
    for _ in range(2000):
        if time.time() - _T0 > budget:
            break
        if len(a) < 2:
            break
        # pick the largest element to try to move
        m = max(a)
        rest = [v for v in a if v != m]
        diffset_rest = set(_diffs(rest))
        need = set(_TARGETS) - diffset_rest
        # candidate positions for m
        cands = set()
        for v in rest:
            for d in need:
                for x in (v + d, v - d):
                    if x >= 0 and x not in rest:
                        cands.add(x)
        if not cands:
            break
        cands = sorted(cands, key=lambda x: x)
        for x in cands:
            # check collision with rest
            ok = True
            for v in rest:
                if x - v in diffset_rest or (v - x) in diffset_rest:
                    ok = False
                    break
            if not ok:
                continue
            newset = rest + [x]
            if _covers(newset):
                newmax = max(newset)
                if newmax < m:
                    a = newset
                    break
    return a


def construct():
    deadline = _T0 + _DEADLINE - 5.0  # safety margin
    rng = random.Random(12345)
    best = None
    bestmax = None

    # initial deterministic greedy (seed-program style)
    a0 = _greedy(0, rng)
    if a0 is not None and _valid(a0):
        a0 = _remove_pass(a0, rng)
        best = a0
        bestmax = _best(a0)

    # randomized greedy starts + removal + shrink
    for it in range(400):
        if time.time() > deadline:
            break
        seed = it + 1
        a = _greedy(seed, rng)
        if a is None or not _is_sidon(a) or not _covers(a):
            continue
        a = _remove_pass(a, rng)
        if time.time() > deadline:
            break
        a = _local_shrink(a, rng, deadline)
        if _valid(a):
            m = _best(a)
            if bestmax is None or m < bestmax:
                bestmax = m
                best = list(a)

    # safety fallback: guarantee a valid result
    if best is None:
        # build a guaranteed valid set via simple construction
        a = [0]
        for d in _TARGETS:
            placed = False
            for v in a:
                for x in (v + d, v - d):
                    if x >= 0 and x not in a and _is_sidon(a + [x]):
                        a.append(x)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                y = 2 * max(a) + 3 * d + 1
                while not _is_sidon(a + [y]):
                    y += 1
                a.append(y)
        best = a

    return best


if __name__ == "__main__":
    print(construct())
