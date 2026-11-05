"""Independent set on the square-sum conflict graph over [1,256], including self-pairs (a==a)."""
import math
import random
import time

# Precompute conflict structure (constant work, tiny).
N = 256
SQUARES = set()
for k in range(1, 23):  # 22^2 = 484 > 512
    SQUARES.add(k * k)

# forbidden[a][b] for a <= b
forbidden = [None] * (N + 1)
for a in range(1, N + 1):
    row = [False] * (N + 1)
    for b in range(a, N + 1):
        if (a + b) in SQUARES:
            row[b] = True
    forbidden[a] = row

# Build adjacency (symmetric) and degrees
adj = [set() for _ in range(N + 1)]
for a in range(1, N + 1):
    for b in range(a, N + 1):
        if forbidden[a][b]:
            adj[a].add(b)
            if b != a:
                adj[b].add(a)

# Self-conflicts: a+a square => a cannot be in set
self_conflict = [False] * (N + 1)
for a in range(1, N + 1):
    if (2 * a) in SQUARES:
        self_conflict[a] = True

# Candidate nodes (no self-conflict)
candidates = [a for a in range(1, N + 1) if not self_conflict[a]]

def _is_square(x):
    if x < 0:
        return False
    r = int(math.isqrt(x))
    return r * r == x

def _valid(s):
    """Check s is a valid subset (all a<=b sums non-square)."""
    arr = sorted(s)
    for i in range(len(arr)):
        a = arr[i]
        if (2 * a) in SQUARES:
            return False
        for j in range(i + 1, len(arr)):
            b = arr[j]
            if (a + b) in SQUARES:
                return False
    return True

def _greedy(order):
    """Greedy add nodes in given order if they don't conflict with chosen."""
    chosen = set()
    for a in order:
        if self_conflict[a]:
            continue
        ok = True
        for b in chosen:
            if a <= b:
                if forbidden[a][b]:
                    ok = False
                    break
            else:
                if forbidden[b][a]:
                    ok = False
                    break
        if ok:
            chosen.add(a)
    return chosen

def _local_search(s, deadline, rng, max_iters=200000):
    """Randomized add/remove local search. Returns best set found."""
    s = set(s)
    best = set(s)
    best_len = len(s)
    iters = 0
    while time.time() < deadline and iters < max_iters:
        iters += 1
        # Try random add
        if candidates:
            a = rng.choice(candidates)
            if a not in s:
                ok = True
                for b in s:
                    if a <= b:
                        if forbidden[a][b]:
                            ok = False
                            break
                    else:
                        if forbidden[b][a]:
                            ok = False
                            break
                if ok:
                    s.add(a)
                    if len(s) > best_len:
                        best_len = len(s)
                        best = set(s)
                    # occasional backtrack: with small prob remove a random element to explore
                    if rng.random() < 0.15:
                        rem = rng.choice(list(s))
                        s.discard(rem)
                    continue
        # Try random remove (helps escape)
        if s and rng.random() < 0.35:
            rem = rng.choice(list(s))
            s.discard(rem)
            continue
        # Fallback: random remove
        if s:
            rem = rng.choice(list(s))
            s.discard(rem)
    if len(s) > best_len:
        best = set(s)
    return best

def construct():
    start = time.time()
    deadline = start + 75.0  # safety margin under 90s kill
    rng = random.Random(12345)

    # Baseline greedy by ascending degree
    deg = [len(adj[a]) for a in range(N + 1)]
    order = sorted(candidates, key=lambda a: (deg[a], a))
    best = _greedy(order)
    best_len = len(best)

    # Multiple randomized greedy restarts with different orderings
    for it in range(200):
        if time.time() > deadline - 20.0:
            break
        # mix: degree-order with random shuffle among ties / random perturb
        if it % 3 == 0:
            o = list(candidates)
            o.sort(key=lambda a: (deg[a] + rng.random() * 5, a))
        elif it % 3 == 1:
            o = list(candidates)
            rng.shuffle(o)
            o.sort(key=lambda a: deg[a])
        else:
            o = list(candidates)
            o.sort(key=lambda a: (deg[a] + rng.random() * 20, a))
        cur = _greedy(o)
        if len(cur) > best_len:
            best_len = len(cur)
            best = cur

    # Local search from best
    best = _local_search(best, deadline, rng)
    best_len = len(best)

    # Additional local-search restarts from diverse greedy starts if time remains
    while time.time() < deadline - 5.0:
        o = list(candidates)
        o.sort(key=lambda a: (deg[a] + rng.random() * 10, a))
        cur = _greedy(o)
        cur = _local_search(cur, deadline, rng, max_iters=50000)
        if len(cur) > best_len:
            best_len = len(cur)
            best = cur

    # Final safety: ensure nonempty and valid
    if not best:
        best = {candidates[0]}
    # Validate
    if not _valid(best):
        # rebuild a safe greedy
        best = _greedy(order)
    return list(best)
