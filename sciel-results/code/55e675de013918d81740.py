"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 150.0
_MARGIN = 3.0

def _valid(s):
    """Sidon check: all positive pairwise differences distinct; covers 1..30; has 0."""
    if 0 not in s:
        return False
    n = len(s)
    if n > 32:
        return False
    if max(s) > 10**6:
        return False
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = s[j] - s[i]
            if d <= 0:
                return False
            if d in seen:
                return False
            seen.add(d)
    for d in range(1, 31):
        if d not in seen:
            return False
    return True

def _rand_set(D, rng, max_elems=12):
    """Random subset of {0..D} containing 0, size ~ max_elems, biased toward small values."""
    # Pick 0 plus random small elements, dedup
    s = {0}
    # Try to get enough elements within D
    attempts = 0
    while len(s) < max_elems and attempts < 200:
        v = rng.randint(1, D)
        s.add(v)
        attempts += 1
    return sorted(s)

def _local_shrink(s, rng, deadline):
    """Try to reduce max element by removing/adjusting elements while staying valid."""
    s = sorted(set(s))
    improved = True
    while improved and time.time() < deadline - _MARGIN:
        improved = False
        # Try removing each non-zero element and re-adding a smaller one
        for i in range(len(s)):
            if s[i] == 0:
                continue
            trial = [x for x in s if x != s[i]]
            if len(trial) < 2:
                continue
            # Try to fill the gap with a smaller element
            D = max(s)
            for _ in range(50):
                v = rng.randint(1, D - 1)
                if v in trial:
                    continue
                t = sorted(trial + [v])
                if _valid(t):
                    s = t
                    improved = True
                    break
        # Try replacing largest element with a smaller one
        if time.time() < deadline - _MARGIN:
            largest = s[-1]
            for _ in range(100):
                v = rng.randint(1, largest - 1)
                if v in s:
                    continue
                t = sorted([x for x in s if x != largest] + [v])
                if _valid(t):
                    s = t
                    improved = True
                    break
    return s

def construct():
    start = time.time()
    deadline = start + _DEADLINE
    rng = random.Random(42)

    best = None
    best_max = 10**6

    # Phase 1: randomized search with decreasing diameter bound
    # Start with a reasonable upper bound and decrease
    D_start = 500
    D_step = 20

    D = D_start
    while D >= 30 and time.time() < deadline - _MARGIN:
        found_in_phase = False
        # Multiple attempts at this diameter
        for attempt in range(200):
            if time.time() >= deadline - _MARGIN:
                break
            s = _rand_set(D, rng, max_elems=12)
            if _valid(s):
                m = max(s)
                if best is None or m < best_max:
                    best = s
                    best_max = m
                    found_in_phase = True
                # Try local shrink
                if time.time() < deadline - _MARGIN:
                    shrunken = _local_shrink(list(s), rng, deadline)
                    m2 = max(shrunken)
                    if m2 < best_max:
                        best = shrunken
                        best_max = m2
                        found_in_phase = True
        if found_in_phase:
            # Decrease diameter since we found something
            D -= D_step
        else:
            # No valid set at this D, keep trying a bit longer then decrease
            D -= D_step // 2

    # Phase 2: more aggressive local optimization on best
    if best is not None and time.time() < deadline - _MARGIN:
        for _ in range(500):
            if time.time() >= deadline - _MARGIN:
                break
            s = list(best)
            # Random perturbation: swap one element
            idx = rng.randint(0, len(s) - 1)
            if s[idx] == 0:
                continue
            old = s[idx]
            new = rng.randint(1, best_max - 1)
            if new in s:
                continue
            s[idx] = new
            s = sorted(s)
            if _valid(s):
                if max(s) < best_max:
                    best = s
                    best_max = max(s)

    # Phase 3: try to reduce size (fewer elements = easier to shrink)
    if best is not None and time.time() < deadline - _MARGIN:
        s = list(best)
        # Try removing elements one at a time
        for i in range(len(s) - 1, -1, -1):
            if s[i] == 0:
                continue
            trial = [x for x in s if x != s[i]]
            if len(trial) >= 2 and _valid(trial):
                s = trial
        best = s
        best_max = max(s)

    if best is None:
        # Fallback: return a trivially valid large set
        # Use the known structure: {0, 1, 3, 4, 9, 11, 12, 17, 19, 20, 25, 27}
        # Check if this covers 1..30
        fallback = [0, 1, 3, 4, 9, 11, 12, 17, 19, 20, 25, 27]
        if _valid(fallback):
            return fallback
        # Ultimate fallback: construct via the original greedy (slow but works)
        a = [0]
        def diffs(ss):
            return [y - x for i, x in enumerate(ss) for y in ss[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            placed = False
            for x in sorted(set(v + dd for v in a for dd in (d, -d) if v + dd >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    placed = True
                    break
            if not placed:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a

    return best
# EVOLVE-BLOCK-END
