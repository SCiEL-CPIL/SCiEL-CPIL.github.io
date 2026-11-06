"""Rewritten: maximum independent set in the 3-uniform hypergraph of equal-GCD triples."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations
import random

def _build():
    N = 128
    g = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            g[i][j] = gcd(i + 1, j + 1)
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            gij = g[i][j]
            for k in range(j + 1, N):
                if g[i][k] == gij and g[j][k] == gij:
                    edges.append((i, j, k))
    return N, g, edges

def _pair_conflict(edges, N):
    # pc[a][b] = bitset of c such that {a,b,c} is a forbidden triple
    pc = [[0] * N for _ in range(N)]
    for (i, j, k) in edges:
        pc[i][j] |= 1 << k
        pc[j][i] |= 1 << k
        pc[i][k] |= 1 << j
        pc[k][i] |= 1 << j
        pc[j][k] |= 1 << i
        pc[k][j] |= 1 << i
    return pc

def construct():
    t0 = time.time()
    DEADLINE = 83.0  # safety margin under the 90s kill
    N, g, edges = _build()
    pc = _pair_conflict(edges, N)

    best_set = None
    best_size = 0

    def greedy(order, start=None):
        # Fast greedy: maintain `blocked` bitset of vertices that would form a
        # forbidden triple with two already-chosen elements.
        chosen = list(start) if start else []
        blocked = 0
        for a, b in combinations(chosen, 2):
            blocked |= pc[a][b]
        for v in order:
            if (blocked >> v) & 1:
                continue
            chosen.append(v)
            for a in chosen[:-1]:
                blocked |= pc[a][v]
        return chosen

    rng = random.Random(12345)

    # conflict-degree of each vertex (how many forbidden triples it appears in)
    deg = [0] * N
    for (i, j, k) in edges:
        deg[i] += 1
        deg[j] += 1
        deg[k] += 1

    fixed_orders = [
        list(range(N)),
        list(range(N - 1, -1, -1)),
        sorted(range(N), key=lambda v: deg[v]),
        sorted(range(N), key=lambda v: deg[v], reverse=True),
        sorted(range(N), key=lambda v: (deg[v], v)),
        sorted(range(N), key=lambda v: (-deg[v], v)),
        sorted(range(N), key=lambda v: (deg[v], -v)),
        sorted(range(N), key=lambda v: (-deg[v], -v)),
    ]
    for o in fixed_orders:
        sol = greedy(o)
        if len(sol) > best_size:
            best_size = len(sol)
            best_set = set(sol)

    # Many randomized greedy restarts (each is O(n^2) bit ops, very cheap).
    while time.time() < DEADLINE - 22:
        o = list(range(N))
        rng.shuffle(o)
        sol = greedy(o)
        if len(sol) > best_size:
            best_size = len(sol)
            best_set = set(sol)

    # Local improvement: remove one element from the current best, then
    # re-greedy the remaining candidates to try to grow the set.
    if best_set is not None:
        base = list(best_set)
        in_best = best_set
        while time.time() < DEADLINE - 5:
            improved = False
            for rem in base:
                if time.time() > DEADLINE - 5:
                    break
                start = [x for x in base if x != rem]
                remaining = [x for x in range(N) if x not in in_best and x != rem]
                rng.shuffle(remaining)
                sol = greedy(remaining, start=start)
                if len(sol) > best_size:
                    best_size = len(sol)
                    best_set = set(sol)
                    base = list(best_set)
                    in_best = best_set
                    improved = True
                    break
            if not improved:
                # second pass with a fresh shuffle of the remove-order
                order_rem = list(base)
                rng.shuffle(order_rem)
                for rem in order_rem:
                    if time.time() > DEADLINE - 5:
                        break
                    start = [x for x in base if x != rem]
                    remaining = [x for x in range(N) if x not in in_best and x != rem]
                    rng.shuffle(remaining)
                    sol = greedy(remaining, start=start)
                    if len(sol) > best_size:
                        best_size = len(sol)
                        best_set = set(sol)
                        base = list(best_set)
                        in_best = best_set
                        improved = True
                        break
                if not improved:
                    break

    if best_set is None:
        best_set = set(range(1, N + 1))

    return list(best_set)
# EVOLVE-BLOCK-END
