"""Simulated annealing for compact Sidon sets covering differences 1..30."""
import random
import time
import math

_DEADLINE = 150.0
_MARGIN = 2.0  # safety margin seconds

def _differences(a):
    n = len(a)
    out = []
    for i in range(n):
        ai = a[i]
        for j in range(i + 1, n):
            out.append(a[j] - ai)
    return out

def _cost(a):
    """Lower is better. Prioritize small max, then coverage, then Sidon-ness."""
    if not a:
        return 10**9
    diffs = _differences(a)
    diffset = set(diffs)
    dup_count = len(diffs) - len(diffset)
    missing = 0
    for d in range(1, 31):
        if d not in diffset:
            missing += 1
    max_elem = a[-1]
    return 10000 * max_elem + 1000 * missing + 100 * dup_count + max_elem

def _valid(a):
    diffs = _differences(a)
    if len(diffs) != len(set(diffs)):
        return False
    s = set(diffs)
    for d in range(1, 31):
        if d not in s:
            return False
    return True

def construct():
    start = time.time()
    deadline = start + _DEADLINE - _MARGIN
    rng = random.Random(12345)

    # Greedy seed: valid but large set covering 1..30.
    a = [0]
    def _diffs_list(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in set(_diffs_list(a)):
            continue
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            dl = _diffs_list(trial)
            if len(set(dl)) == len(dl):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                dl = _diffs_list(trial)
                if len(set(dl)) == len(dl):
                    a = trial
                    break
                y += 1
    a = sorted(set(a))
    best_a = a[:]
    best_cost = _cost(a)

    T_min = 1e-4
    cooling = 0.9999

    def _exp(x):
        if x < -50:
            return 0.0
        return math.exp(x)

    def _try_move(cur, cur_cost, T):
        r = rng.random()
        n = len(cur)
        if r < 0.35:
            if n <= 1:
                return cur, cur_cost
            idx = rng.randint(1, n - 1)
            old = cur[idx]
            new = max(0, old + rng.randint(-300, 300))
            if new == old:
                return cur, cur_cost
            trial = cur[:]
            trial[idx] = new
            if len(set(trial)) != len(trial):
                return cur, cur_cost
            trial = sorted(trial)
            nc = _cost(trial)
            d = nc - cur_cost
            if d <= 0 or rng.random() < min(1.0, _exp(-d / max(T, 1e-9))):
                return trial, nc
            return cur, cur_cost
        elif r < 0.65:
            hi = max(cur) + 100
            new = rng.randint(1, hi)
            if new in cur:
                return cur, cur_cost
            trial = sorted(cur + [new])
            nc = _cost(trial)
            d = nc - cur_cost
            if d <= 0 or rng.random() < min(1.0, _exp(-d / max(T, 1e-9))):
                return trial, nc
            return cur, cur_cost
        else:
            if n <= 1:
                return cur, cur_cost
            idx = rng.randint(1, n - 1)
            trial = cur[:idx] + cur[idx + 1:]
            nc = _cost(trial)
            d = nc - cur_cost
            if d <= 0 or rng.random() < min(1.0, _exp(-d / max(T, 1e-9))):
                return trial, nc
            return cur, cur_cost

    for _restart in range(100):
        if time.time() > deadline:
            break
        cur = best_a[:]
        cur_cost = best_cost
        T = max(1.0, best_cost * 0.05)
        for _ in range(200000):
            if time.time() > deadline:
                break
            cur, cur_cost = _try_move(cur, cur_cost, T)
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_a = cur[:]
            T *= cooling
            if T < T_min:
                T = T_min

    return best_a
