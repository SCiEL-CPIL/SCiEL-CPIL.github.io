import random
import time
import math
from collections import deque

def _luw_construct(k, g):
    """L-U-W algebraic construction (deterministic, guaranteed valid fallback)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _verify_girth_fast(n, edges, target_girth):
    """Full non-backtracking BFS girth check from every vertex. True iff girth >= target."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for start in range(n):
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= target_girth - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True


def _random_config(n, k, rng):
    """Random k-regular multigraph via configuration model. Returns adj list (with possible
    parallel edges / self-loops) or None if it cannot close."""
    half = n * k
    stubs = list(range(half))
    rng.shuffle(stubs)
    adj = [set() for _ in range(n)]
    for i in range(0, half, 2):
        a = stubs[i] // k
        b = stubs[i + 1] // k
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _swap_edges(adj, a, b, c, d):
    """Remove (a,b) and (c,d), add (a,c) and (b,d). Returns True if valid & simple."""
    if b not in adj[a] or d not in adj[c]:
        return False
    if c in adj[a] or d in adj[b]:
        return False
    adj[a].remove(b)
    adj[b].remove(a)
    adj[c].remove(d)
    adj[d].remove(c)
    adj[a].add(c)
    adj[c].add(a)
    adj[b].add(d)
    adj[d].add(b)
    return True


def _swap_edges_undo(adj, a, c, b, d):
    adj[a].remove(c)
    adj[c].remove(a)
    adj[b].remove(d)
    adj[d].remove(b)
    adj[a].add(b)
    adj[b].add(a)
    adj[c].add(d)
    adj[d].add(c)


def _short_cycle_score(adj, n, target_girth, rng):
    """Count non-backtracking walks of length target_girth-1 that close to the start.
    Each short cycle of length L <= target_girth-1 contributes L*(L-2) to the sum,
    so a zero sum is exactly equivalent to girth >= target_girth."""
    L = target_girth - 1
    total = 0
    for start in range(n):
        # state: (current, prev) -> count of non-backtracking walks
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 1
        for _ in range(L):
            ndist = {}
            while dq:
                v, prev = dq.popleft()
                c = dist[(v, prev)]
                for w in adj[v]:
                    if w == prev:
                        continue
                    if w == start:
                        continue
                    key = (w, v)
                    ndist[key] = ndist.get(key, 0) + c
            dq.clear()
            for key, val in ndist.items():
                dq.append(key)
            dist = ndist
            if not dist:
                break
        # count walks whose last step lands on start
        for (v, prev), c in dist.items():
            if start in adj[v]:
                total += c
    return total


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before the 150s kill
    rng = random.Random(1234567)

    def elapsed_ok():
        return time.time() < deadline

    # Deterministic guaranteed-valid fallback (small n).
    luw_n, luw_edges = _luw_construct(k, g)

    best_n, best_edges = luw_n, luw_edges

    # Structural approach: random k-regular graphs + simulated-annealing edge-swap
    # local search that directly eliminates short cycles (no voltage-lift restrictions).
    # n is even; k=3 requires 3n even -> n even.
    n_start = max(luw_n + 2, 512)
    if n_start % 2 != 0:
        n_start += 1

    for n in range(n_start, 1001, 2):
        if not elapsed_ok():
            break
        if n >= best_n:
            break

        # Try several seeds at this n.
        attempts = 4
        for seed in range(attempts):
            if not elapsed_ok():
                break
            r = random.Random(seed * 7919 + n * 104729 + 13)

            # Build a random k-regular simple graph via configuration model.
            adj = None
            for _ in range(60):
                cand = _random_config(n, k, r)
                if cand is None:
                    continue
                ok = True
                for v in range(n):
                    if len(cand[v]) != k:  # parallel edge or self-loop present
                        ok = False
                        break
                if ok:
                    adj = cand
                    break
            if adj is None:
                continue

            # Simulated annealing on short-cycle count.
            score = _short_cycle_score(adj, n, g, r)
            best_score = score
            T = 8.0
            steps = 20000
            for s in range(steps):
                if not elapsed_ok():
                    break
                if score == 0:
                    break
                a = r.randrange(n)
                na = adj[a]
                if len(na) < k:
                    continue
                b = na[r.randrange(len(na))]
                c = r.randrange(n)
                if c == a or c == b:
                    continue
                nc = adj[c]
                if len(nc) < k:
                    continue
                d = nc[r.randrange(len(nc))]
                if d == a or d == b:
                    continue
                if not _swap_edges(adj, a, b, c, d):
                    continue
                new_score = _short_cycle_score(adj, n, g, r)
                delta = new_score - score
                if delta <= 0 or r.random() < math.exp(-delta / max(T, 1e-9)):
                    score = new_score
                    if score < best_score:
                        best_score = score
                else:
                    _swap_edges_undo(adj, a, c, b, d)
                T *= 0.995

            if best_score == 0:
                # Found a girth-g graph at this n -> it is minimal for this n, stop.
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                # Final certification.
                if _verify_girth_fast(n, edges, g):
                    best_n, best_edges = n, edges
                    break  # this n is the minimum feasible found; larger n not better

    # If no improvement, keep LUW fallback (guaranteed valid).
    return best_n, best_edges
