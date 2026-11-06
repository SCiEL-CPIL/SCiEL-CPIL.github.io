"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

_DEADLINE = 150.0
_MARGIN = 3.0

def _diffs_set(s):
    """Return set of all positive pairwise differences of sorted list s."""
    n = len(s)
    out = set()
    for i in range(n):
        for j in range(i + 1, n):
            out.add(s[j] - s[i])
    return out

def _valid(s):
    """Sidon check: all positive pairwise differences distinct; covers 1..30; has 0; size <= 32."""
    if 0 not in s:
        return False
    n = len(s)
    if n > 32 or n < 2:
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

def _energy(s):
    """Composite energy: collision penalty + uncovered penalty + diameter penalty."""
    n = len(s)
    seen = set()
    collisions = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = s[j] - s[i]
            if d <= 0:
                collisions += 10
            elif d in seen:
                collisions += 5
            else:
                seen.add(d)
    uncovered = sum(1 for d in range(1, 31) if d not in seen)
    diameter = max(s) if s else 0
    return 100 * collisions + 20 * uncovered + diameter

def _greedy_construct(target_D, rng, max_iter=200):
    """Greedy constructive: build set covering 1..30 within target_D."""
    a = [0]
    needed = set(range(1, 31))
    for _ in range(max_iter):
        if time.time() > _DEADLINE - _MARGIN:
            break
        if not needed:
            break
        best_x = None
        best_gain = 0
        candidates = list(range(1, target_D + 1))
        rng.shuffle(candidates)
        # Limit search for speed
        for x in candidates[:300]:
            if x in a:
                continue
            trial = sorted(a + [x])
            seen = set()
            coll = 0
            new_cov = 0
            for i in range(len(trial)):
                for j in range(i + 1, len(trial)):
                    d = trial[j] - trial[i]
                    if d in seen:
                        coll += 1
                    else:
                        seen.add(d)
                        if d in needed:
                            new_cov += 1
            if coll == 0 and new_cov > best_gain:
                best_gain = new_cov
                best_x = x
        if best_x is not None and best_gain > 0:
            a = sorted(a + [best_x])
            seen = _diffs_set(a)
            needed = needed - seen
        else:
            break
    return a

def _local_sa(s, rng, deadline, max_iter=500, T0=50.0, cooling=0.995):
    """Simulated-annealing local refinement."""
    s = list(s)
    cur_e = _energy(s)
    T = T0
    best_s = list(s)
    best_e = cur_e
    for _ in range(max_iter):
        if time.time() > deadline - _MARGIN:
            break
        idx = rng.randint(1, len(s) - 1)
        old_val = s[idx]
        if rng.random() < 0.6:
            delta = rng.randint(-30, 30)
            new_val = old_val + delta
        else:
            new_val = rng.randint(1, max(s))
        if new_val <= 0 or new_val in s:
            continue
        s[idx] = new_val
        s.sort()
        new_e = _energy(s)
        if new_e < cur_e or (new_e > cur_e and rng.random() < math.exp(-(new_e - cur_e) / max(T, 0.1))):
            cur_e = new_e
            if new_e < best_e:
                best_e = new_e
                best_s = list(s)
        else:
            s[idx] = old_val
            s.sort()
        T *= cooling
    return best_s

def _rand_initial(D, rng, size=12):
    """Random initial set containing 0, biased toward small values."""
    s = {0}
    for _ in range(size - 1):
        v = int(rng.random() ** 0.7 * D) + 1
        if v not in s:
            s.add(v)
    return sorted(s)

def _local_shrink(s, rng, deadline):
    """Try to reduce max element by removing/adjusting elements while staying valid."""
    s = sorted(set(s))
    improved = True
    while improved and time.time() < deadline - _MARGIN:
        improved = False
        for i in range(len(s)):
            if s[i] == 0:
                continue
            trial = [x for x in s if x != s[i]]
            if len(trial) < 2:
                continue
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

def _try_remove_elements(s, rng, deadline):
    """Try removing elements one at a time while staying valid."""
    s = list(s)
    changed = True
    while changed and time.time() < deadline - _MARGIN:
        changed = False
        for i in range(len(s) - 1, 0, -1):
            if s[i] == 0:
                continue
            trial = [x for x in s if x != s[i]]
            if len(trial) >= 2 and _valid(trial):
                s = trial
                changed = True
                break
    return s

def construct():
    start = time.time()
    deadline = start + _DEADLINE
    rng = random.Random(42)

    best = None
    best_max = 10**6

    # Phase 1: Multiple randomized greedy constructions with varying D
    for D in [60, 80, 100, 120, 150, 200, 300, 500]:
        if time.time() > deadline - _MARGIN:
            break
        for attempt in range(20):
            if time.time() > deadline - _MARGIN:
                break
            s = _greedy_construct(D, rng)
            if len(s) >= 2 and _valid(s):
                m = max(s)
                if m < best_max:
                    best = list(s)
                    best_max = m
                # SA refinement
                refined = _local_sa(list(s), rng, deadline, max_iter=400)
                if _valid(refined):
                    m2 = max(refined)
                    if m2 < best_max:
                        best = list(refined)
                        best_max = m2

    # Phase 2: Random initial + SA
    for D in [50, 70, 90, 120, 150, 200]:
        if time.time() > deadline - _MARGIN:
            break
        for attempt in range(40):
            if time.time() > deadline - _MARGIN:
                break
            s = _rand_initial(D, rng, size=12)
            refined = _local_sa(list(s), rng, deadline, max_iter=1000)
            if _valid(refined):
                m = max(refined)
                if m < best_max:
                    best = list(refined)
                    best_max = m

    # Phase 3: Aggressive local optimization on best
    if best is not None:
        if time.time() < deadline - _MARGIN:
            refined = _local_sa(list(best), rng, deadline, max_iter=3000, T0=80.0, cooling=0.998)
            if _valid(refined):
                m = max(refined)
                if m < best_max:
                    best = list(refined)
                    best_max = m

        # Try element removal
        if time.time() < deadline - _MARGIN:
            s = _try_remove_elements(list(best), rng, deadline)
            if _valid(s):
                m = max(s)
                if m < best_max:
                    best = s
                    best_max = m

        # Local shrink
        if time.time() < deadline - _MARGIN:
            shrunken = _local_shrink(list(best), rng, deadline)
            if _valid(shrunken):
                m = max(shrunken)
                if m < best_max:
                    best = shrunken
                    best_max = m

        # Final SA pass
        if time.time() < deadline - _MARGIN:
            refined = _local_sa(list(best), rng, deadline, max_iter=2000, T0=30.0, cooling=0.997)
            if _valid(refined):
                m = max(refined)
                if m < best_max:
                    best = list(refined)
                    best_max = m

    if best is None:
        # Fallback: greedy construct
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
