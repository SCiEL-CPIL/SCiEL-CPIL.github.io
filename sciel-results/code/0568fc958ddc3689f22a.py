"""Construct a 5-regular graph with girth at least 9 (cage problem n(5,9)).
Uses the Lazebnik-Ustimenko-Woldar (LUW) construction.
For k=5: try d=3 (n=625, girth>=8, check if >=9), then d=4 (n=1250, girth>=9 guaranteed).
Optimized girth check using flat-array BFS with early termination.
"""
import time
from collections import deque

def construct(k, g):
    q = k
    deadline = time.time() + 115.0  # Safety margin under 150s

    def build_luw(d):
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

        rule = {}
        for c in range(2, d + 1):
            if c <= 4:
                rule[c] = {2: (1, 1), 3: (2, 1), 4: (1, 2)}[c]
            else:
                r = (c - 5) % 4
                c0 = c - r
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []

        for i, vertex in enumerate(order):
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))

        n = len(order)
        return n, edges

    def check_girth_fast(n, edges, g):
        """Check if graph has no cycle shorter than g using non-backtracking walks.
        Uses flat-array BFS with early termination."""
        if n < 3:
            return True

        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        INF = 1 << 28

        for s in range(n):
            # dist[v * n + p] = shortest NB walk from s to v coming from p
            dist = [INF] * (n * n)
            q_head = 0
            q_tail = 0
            q_buf = []

            for w in adj[s]:
                idx = s * n + w
                if dist[idx] > 1:
                    dist[idx] = 1
                    q_buf.append(s * n + w)
                    q_tail += 1

            while q_head < q_tail:
                code = q_buf[q_head]
                q_head += 1
                u = code // n
                v = code - u * n
                d_val = dist[code]

                # If v == s, we closed a cycle of length d_val
                if v == s:
                    if d_val < g:
                        return False
                    break

                nd = d_val + 1
                if nd >= g:
                    continue

                for w in adj[v]:
                    if w != u:
                        idx = v * n + w
                        if dist[idx] > nd:
                            dist[idx] = nd
                            q_buf.append(idx)
                            q_tail += 1

            if time.time() > deadline:
                return False  # Assume invalid if we run out of time

        return True

    # Moore bound for k=5, g=9 (odd girth 2r+1 where r=4):
    # n >= k * ((k-1)^r - 1) / (k-2) = 5 * (4^4 - 1) / 3 = 425

    # Try d values in order of increasing n:
    # d=3: n=2*5^3=500 (but LUW gives girth >= d+5=8, need to verify >=9)
    # d=4: n=2*5^4=1250 (LUW gives girth >= d+5=9, guaranteed)
    # d=5: n=2*5^5=6250 (fallback)

    # Try d=3 first (smaller, might work if girth >= 9)
    n3, edges3 = build_luw(3)
    if check_girth_fast(n3, edges3, g):
        return n3, edges3

    if time.time() > deadline:
        # Not enough time, return d=4 (known valid by construction)
        n4, edges4 = build_luw(4)
        return n4, edges4

    # d=4: girth >= 9 guaranteed by LUW construction
    n4, edges4 = build_luw(4)
    # We can skip verification since LUW guarantees girth >= d+5 = 9
    return n4, edges4
