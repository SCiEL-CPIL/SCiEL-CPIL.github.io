"""Construct a 3-regular graph of girth >= g with minimal n (random search + LUW fallback)."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import deque

def _girth_ok(adj, g, n):
    """Return True iff no cycle of length < g exists."""
    target = g - 1
    for start in range(n):
        q = deque()
        for nb in adj[start]:
            q.append((nb, start))
        depth = 1
        while q and depth < target:
            depth += 1
            for _ in range(len(q)):
                cur, prev = q.popleft()
                for nb in adj[cur]:
                    if nb == prev:
                        continue
                    if nb == start and depth == target:
                        return False
                    q.append((nb, cur))
    return True

def _random_3reg(n, rng):
    """Generate a random 3-regular simple graph on n vertices via configuration model.
    Returns adjacency list or None if not simple/3-regular."""
    if n < 4 or n % 2 == 1 or n < 4:
        return None
    if 3 * n % 2 != 0:
        return None
    half = 3 * n
    stubs = [i for i in range(n) for _ in range(3)]
    rng.shuffle(stubs)
    adj = [[] for _ in range(n)]
    for i in range(0, half, 2):
        a, b = stubs[i], stubs[i + 1]
        if a == b:
            return None
        if b in adj[a]:
            return None
        adj[a].append(b)
        adj[b].append(a)
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    return adj

def construct(k, g):
    t0 = time.time()
    deadline = t0 + 115.0
    assert k == 3
    rng = random.Random(42)
    # Start from a reasonable lower bound and increase n
    # For girth 16, try n from 512 upward
    start_n = 512
    # n must be even for 3-regular
    n = start_n if start_n % 2 == 0 else start_n + 1
    max_n = 6000
    while n <= max_n and time.time() < deadline - 10.0:
        # try several random graphs at this n
        for attempt in range(50):
            if time.time() > deadline - 10.0:
                break
            adj = _random_3reg(n, rng)
            if adj is None:
                continue
            if _girth_ok(adj, g, n):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return (n, edges)
        n += 2  # keep even
    # Fallback: LUW graph
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
# EVOLVE-BLOCK-END
