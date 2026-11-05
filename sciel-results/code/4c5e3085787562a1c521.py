import time
import random
import math

_TIME_LIMIT = 50  # safety margin before 60s

def _valid_set(S, N, squares, adj):
    """Verify S is a valid independent set: no a<=b in S with a+b square."""
    # quick check: for each a in S, 2a not square, and no neighbor in S
    for a in S:
        if 2 * a in squares:
            return False
        for b in adj[a - 1]:
            if b >= a and b in S:
                return False
    return True

def _improve(S, N, adj, squares, iters=None):
    """Local improvement: try single additions, then 2-swaps. Returns improved set."""
    S = list(S)
    inS = set(S)
    improved = True
    passes = 0
    while improved:
        improved = False
        passes += 1
        if iters is not None and passes > iters:
            break
        # Try adding each non-member whose all neighbors are outside S
        for v in range(1, N + 1):
            if v in inS:
                continue
            ok = True
            for u in adj[v - 1]:
                if u in inS:
                    ok = False
                    break
            if ok:
                S.append(v)
                inS.add(v)
                improved = True
        # Try 2-swaps: remove x, add y and z? Simpler: remove x, add y where y conflicts only with x
        for x in list(S):
            inS.remove(x)
            S.remove(x)
            # try to add y that now becomes free (its only S-neighbor was x)
            for y in range(1, N + 1):
                if y in inS:
                    continue
                ok = True
                for u in adj[y - 1]:
                    if u in inS:
                        ok = False
                        break
                if ok:
                    S.append(y)
                    inS.add(y)
                    improved = True
                    break
            if not improved:
                # put x back
                inS.add(x)
                S.append(x)
    return S

def _greedy_random(N, adj, order):
    S = []
    inS = set()
    for v in order:
        ok = True
        for u in adj[v - 1]:
            if u in inS:
                ok = False
                break
        if ok:
            S.append(v)
            inS.add(v)
    return S

def _greedy_min_deg(N, adj):
    remaining = set(range(1, N + 1))
    S = []
    inS = set()
    while remaining:
        # pick v in remaining with min number of remaining neighbors
        best = None
        best_deg = None
        # sample if large for speed
        for v in remaining:
            d = sum(1 for u in adj[v - 1] if u in remaining)
            if best_deg is None or d < best_deg:
                best_deg = d
                best = v
        S.append(best)
        inS.add(best)
        remaining.discard(best)
        # remove neighbors of best from remaining
        for u in adj[best - 1]:
            remaining.discard(u)
    return S

def construct():
    start = time.time()
    N = 256
    max_sum = 2 * N
    squares = set()
    k = 1
    while k * k <= max_sum:
        squares.add(k * k)
        k += 1

    # adjacency: adj[i-1] = list of j such that i+j is a square
    adj = [[] for _ in range(N)]
    for i in range(1, N + 1):
        for sq in squares:
            u = sq - i
            if 1 <= u <= N:
                adj[i - 1].append(u)
    # dedupe & keep sorted; remove self (i==u) since a=b handled by 2a square check
    for i in range(N):
        s = set(adj[i])
        s.discard(i + 1)
        adj[i] = sorted(s)

    best = None
    best_size = -1

    # Strategy 1: min-degree greedy (deterministic strong baseline)
    s1 = _greedy_min_deg(N, adj)
    s1 = _improve(s1, N, adj, squares)
    if len(s1) > best_size:
        best_size = len(s1)
        best = s1

    # Strategy 2: many randomized greedy orders + improvement
    while time.time() - start < _TIME_LIMIT:
        order = list(range(1, N + 1))
        random.shuffle(order)
        cand = _greedy_random(N, adj, order)
        cand = _improve(cand, N, adj, squares)
        if len(cand) > best_size:
            best_size = len(cand)
            best = cand

        # also min-degree with random tie-break via random order seed
        order2 = list(range(1, N + 1))
        random.shuffle(order2)
        # biased greedy: process in order but pick min-degree among remaining each step
        remaining = set(range(1, N + 1))
        cand2 = []
        inS = set()
        # deterministic min-degree but break ties by order2 rank
        rank = {v: i for i, v in enumerate(order2)}
        while remaining:
            best = None
            best_key = None
            for v in remaining:
                d = sum(1 for u in adj[v - 1] if u in remaining)
                key = (d, rank[v])
                if best_key is None or key < best_key:
                    best_key = key
                    best = v
            cand2.append(best)
            inS.add(best)
            remaining.discard(best)
            for u in adj[best - 1]:
                remaining.discard(u)
        cand2 = _improve(cand2, N, adj, squares)
        if len(cand2) > best_size:
            best_size = len(cand2)
            best = cand2

    # final polish: run improvement again on best
    if best is not None:
        best = _improve(best, N, adj, squares)

    # fallback (should never trigger)
    if best is None or len(best) == 0:
        # simple safe set: numbers with 2a not square and spaced out
        best = [a for a in range(1, N + 1) if 2 * a not in squares][:1]

    # final validity check; if invalid, fall back to a known-safe construction
    if not _valid_set(best, N, squares, adj):
        # build a safe set greedily by value
        safe = []
        inS = set()
        for a in range(1, N + 1):
            if 2 * a in squares:
                continue
            if any(u in inS for u in adj[a - 1]):
                continue
            safe.append(a)
            inS.add(a)
        best = safe if safe else [3]

    return list(best)
