# EVOLVE-BLOCK-START
import time
import random
from math import gcd

_N = 128
_ALL = list(range(1, _N + 1))

# Precompute gcd matrix
_GCD = [[0] * (_N + 1) for _ in range(_N + 1)]
for i in range(1, _N + 1):
    for j in range(i, _N + 1):
        g = gcd(i, j)
        _GCD[i][j] = g
        _GCD[j][i] = g


def _is_valid(S):
    s = list(S)
    n = len(s)
    for i in range(n):
        a = s[i]
        for j in range(i + 1, n):
            b = s[j]
            gab = _GCD[a][b]
            for k in range(j + 1, n):
                c = s[k]
                if gab == _GCD[a][c] == _GCD[b][c]:
                    return False
    return True


def _can_add(x, selected):
    """Can we add x to selected without creating a bad triple?"""
    for i in range(len(selected)):
        a = selected[i]
        gax = _GCD[x][a]
        for j in range(i + 1, len(selected)):
            b = selected[j]
            if gax == _GCD[x][b] == _GCD[a][b]:
                return False
    return True


def _greedy(order, start=None):
    selected = []
    sel_set = set()
    if start:
        selected = list(start)
        sel_set = set(start)
    for x in order:
        if x in sel_set:
            continue
        if _can_add(x, selected):
            selected.append(x)
            sel_set.add(x)
    return selected


def _local_search(S, budget=500):
    """Swap-based local search: remove one, add one excluded."""
    S = list(S)
    sset = set(S)
    improved = True
    while improved and budget > 0:
        improved = False
        excluded = [x for x in _ALL if x not in sset]
        for i in range(len(S)):
            if budget <= 0:
                break
            rem = S[i]
            new_S = S[:i] + S[i + 1:]
            for add in excluded:
                if add in sset:
                    continue
                trial = new_S + [add]
                if _is_valid(trial):
                    S = trial
                    sset = set(S)
                    improved = True
                    budget -= 1
                    break
            if improved:
                break
    return S


def _num_divisors(x):
    cnt = 0
    d = 2
    t = x
    while d * d <= t:
        if t % d == 0:
            cnt += 2
            while t % d == 0:
                t //= d
        d += 1
    if t > 1:
        cnt += 1
    return cnt


def _is_prime(n):
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += 1
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin

    best = []
    best_len = 0

    rng = random.Random(12345)

    # Deterministic baselines
    for order in [_ALL,
                  list(reversed(_ALL)),
                  sorted(_ALL, key=lambda x: -_num_divisors(x)),
                  sorted(_ALL, key=lambda x: _num_divisors(x)),
                  [x for x in _ALL if _is_prime(x)] + [x for x in _ALL if not _is_prime(x)],
                  [x for x in _ALL if not _is_prime(x)] + [x for x in _ALL if _is_prime(x)]]:
        s = _greedy(order)
        if len(s) > best_len:
            best, best_len = s, len(s)

    # Randomized restarts
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        order = list(_ALL)
        rng.shuffle(order)
        s = _greedy(order)
        if len(s) > best_len:
            best, best_len = s, len(s)
            if time.time() < deadline - 15.0:
                s2 = _local_search(best, budget=200)
                if len(s2) > best_len:
                    best, best_len = s2, len(s2)

        # Random start then complete
        if attempt % 5 == 0 and time.time() < deadline:
            start_size = rng.randint(0, 10)
            start = rng.sample(_ALL, start_size)
            remaining = [x for x in _ALL if x not in set(start)]
            rng.shuffle(remaining)
            s = _greedy(remaining, start=start)
            if len(s) > best_len:
                best, best_len = s, len(s)

    # Final local search
    if time.time() < deadline - 2.0 and best_len > 0:
        s2 = _local_search(best, budget=400)
        if len(s2) > best_len:
            best, best_len = s2, len(s2)

    # Verify
    if not best or not _is_valid(best):
        best = [1]

    return best
# EVOLVE-BLOCK-END
