import time
import math
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = 145.0  # safety margin below 150s wall-clock

    def elapsed():
        return time.time() - start_time

    def moore_bound(k, g):
        if k <= 1:
            return 0
        if g == 1:
            return 1
        r = g // 2
        if g % 2 == 1:
            s = 1
            term = k
            for _ in range(r - 1):
                s += term
                term *= (k - 1)
            return s
        else:
            s = 0
            term = 1
            for _ in range(r):
                s += term
                term *= (k - 1)
            return 2 * s

    n_min = moore_bound(k, g)

    def girth_of(adj, n):
        best = float('inf')
        for u in range(n):
            dist = {}
            q = deque()
            for v in adj[u]:
                dist[(v, u)] = 1
                q.append((v, u))
            while q:
                v, parent = q.popleft()
                d = dist[(v, parent)]
                for w in adj[v]:
                    if w == parent:
                        continue
                    if w == u:
                        if d + 1 < best:
                            best = d + 1
                        continue
                    state = (w, v)
                    if state not in dist:
                        dist[state] = d + 1
                        q.append(state)
        return best

    def is_valid(k, g, n, edges):
        adj = [set() for _ in range(n)]
        seen = set()
        for (i, j) in edges:
            if i == j:
                return False
            if i < 0 or i >= n or j < 0 or j >= n:
                return False
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                return False
            seen.add((a, b))
            adj[i].add(j)
            adj[j].add(i)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        visited = [False] * n
        dq = deque([0])
        visited[0] = True
        cnt = 0
        while dq:
            u = dq.popleft()
            cnt += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    dq.append(v)
        if cnt != n:
            return False
        gr = girth_of(adj, n)
        return gr >= g

    def luw_construct(k, d):
        q = k
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

    best = None
    best_n = float('inf')

    def consider(n, edges):
        nonlocal best, best_n
        if best is None or n < best_n:
            if is_valid(k, g, n, edges):
                best = (n, edges)
                best_n = n
                return True
        return False

    # Try LUW constructions with increasing d
    # n grows as q^d, so small d is preferred
    max_d = 10 # Safety limit
    for d in range(3, max_d + 1):
        if elapsed() > deadline - 10:
            break
        try:
            n, edges = luw_construct(k, d)
            # Optional: quick girth check before full validity?
            # is_valid does full check.
            if consider(n, edges):
                # If we found a valid one, we might still find a smaller one with different d?
                # No, n increases with d. So the first valid d is the smallest n for LUW.
                # But wait, different d might have different connectivity?
                # LUW is connected.
                # So we can break?
                # Actually, smaller d -> smaller n.
                # If d=3 is valid, it's better than d=4.
                # So we can return immediately?
                # Yes, for a fixed k, n(d) is increasing.
                return best
        except Exception:
            pass

    # If LUW failed (e.g. girth too low for all d < max_d), try Morgenstern?
    # Morgenstern n=q+1. For g=9, q must be huge.
    # So LUW is the best bet.
    
    if best is None:
        # Fallback: return a large LUW graph
        try:
            n, edges = luw_construct(k, max_d)
            best = (n, edges)
        except Exception:
            # Last resort
            pass

    return best

# EVOLVE-BLOCK-END
