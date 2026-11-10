"""Search for a hard ambient set: 12 distinct ints in [0,128] whose largest
diagonal-Sidon subset is as small as possible.

A "diagonal-Sidon" subset S has all unordered pair sums unique INCLUDING
diagonals: for any i<=j and k<=l with {i,j} != {k,l}, a[i]+a[j] != a[k]+a[l].
Equivalently: all self-sums {2a} are distinct (true if a distinct), all
cross-sums {a+b, a<b} are distinct, and no cross-sum equals any self-sum.
"""
import random
import time

MAX_VAL = 128
N = 12
DEADLINE = 85.0  # safety margin under 90s wall-clock


def _max_diagonal_sidon(arr):
    """Exact largest diagonal-Sidon subset size of arr (n small)."""
    n = len(arr)
    a = list(arr)
    # Precompute sum value for each unordered pair (i<=j)
    # pair index p = i*n + j
    sumv = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            sumv[i][j] = a[i] + a[j]
    best = 0

    def rec(start, size, cur, used):
        nonlocal best
        # Prune: even taking all remaining can't beat best
        if size + (n - start) <= best:
            return
        if size > best:
            best = size
        for i in range(start, n):
            ok = True
            # Check self-sum 2*a[i] against used
            if used & (1 << (i * n + i)):
                ok = False
            else:
                # Check cross sums a[i]+a[j] for j in cur
                for j in cur:
                    if i == j:
                        continue
                    if used & (1 << (min(i, j) * n + max(i, j))):
                        ok = False
                        break
            if not ok:
                continue
            new_used = used | (1 << (i * n + i))
            for j in cur:
                if i == j:
                    continue
                new_used |= (1 << (min(i, j) * n + max(i, j)))
            cur.append(i)
            rec(i + 1, size + 1, cur, new_used)
            cur.pop()

    rec(0, 0, [], 0)
    return best


def _greedy_build(rng, iters=400):
    """Greedy: add elements that most reduce the current max Sidon size."""
    best_set = None
    best_score = 99
    for _ in range(iters):
        s = set()
        for _ in range(N):
            cands = [v for v in range(MAX_VAL + 1) if v not in s]
            # sample a subset for speed
            if len(cands) > 200:
                sample = rng.sample(cands, 200)
            else:
                sample = cands
            trial = list(s)
            best_v = None
            best_local = 99
            for v in sample:
                trial2 = trial + [v]
                if len(trial2) < N:
                    # For partial sets, use a proxy: count of collisions so far
                    # Actually just compute max sidon of current partial set
                    sc = _max_diagonal_sidon(trial2)
                else:
                    sc = _max_diagonal_sidon(trial2)
                if sc < best_local:
                    best_local = sc
                    best_v = v
            if best_v is None:
                best_v = rng.choice(cands)
            s.add(best_v)
        arr = sorted(s)
        sc = _max_diagonal_sidon(arr)
        if sc < best_score:
            best_score = sc
            best_set = arr
    return best_set, best_score


def _hill_climb(start, rng, deadline):
    """Local search: swap a random element for a random new value."""
    arr = list(start)
    cur_score = _max_diagonal_sidon(arr)
    best_arr = list(arr)
    best_score = cur_score
    t0 = time.time()
    stall = 0
    while time.time() - t0 < deadline:
        i = rng.randrange(N)
        newv = rng.randrange(MAX_VAL + 1)
        if newv in arr:
            continue
        trial = list(arr)
        trial[i] = newv
        sc = _max_diagonal_sidon(trial)
        if sc < cur_score:
            arr = trial
            cur_score = sc
            stall = 0
            if cur_score < best_score:
                best_score = cur_score
                best_arr = list(arr)
        else:
            stall += 1
            # occasional random restart around best
            if stall > 300:
                arr = list(best_arr)
                cur_score = best_score
                stall = 0
    return best_arr, best_score


def construct():
    rng = random.Random(12345)
    t0 = time.time()
    deadline = t0 + DEADLINE

    # Initial: a few greedy builds
    best_arr, best_score = None, 99
    while time.time() < deadline - 20:
        arr, sc = _greedy_build(rng, iters=150)
        if arr is not None and sc < best_score:
            best_score = sc
            best_arr = arr
        if best_score <= 9:  # strong enough, stop building
            break

    if best_arr is None:
        best_arr = list(range(N))
        best_score = _max_diagonal_sidon(best_arr)

    # Hill climb to polish
    arr, sc = _hill_climb(best_arr, rng, deadline - 2)
    if sc < best_score:
        best_score = sc
        best_arr = arr

    return sorted(best_arr)
