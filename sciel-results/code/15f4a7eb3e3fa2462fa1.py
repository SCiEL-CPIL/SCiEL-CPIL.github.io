import time
import math
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety

    best_n = float('inf')
    best_edges = None

    def verify_graph(n, edges, k, g, time_limit=10.0):
        """
        Verify that the graph is:
        1. Simple (no loops, no multi-edges)
        2. k-regular
        3. Has girth >= g (using non-backtracking walk BFS)
        """
        if time.time() > start_time + time_limit:
            return False

        if n == 0:
            return False

        # Check degree and build adjacency list
        adj = [[] for _ in range(n)]
        seen_edges = set()

        for u, v in edges:
            if u == v:
                return False
            if u < 0 or u >= n or v < 0 or v >= n:
                return False
            edge_key = (min(u, v), max(u, v))
            if edge_key in seen_edges:
                return False
            seen_edges.add(edge_key)
            adj[u].append(v)
            adj[v].append(u)

        # Check regularity
        for i in range(n):
            if len(adj[i]) != k:
                return False

        # Girth check using non-backtracking BFS from each vertex
        # For each start vertex, do BFS tracking (node, prev) states
        # A cycle of length L < g exists iff we can return to start
        # within < g/2 non-backtracking steps (for even g) or similar
        max_depth = (g - 1) // 2  # We only need to check up to this depth

        for u in range(n):
            if time.time() > start_time + time_limit:
                return False

            # dist[prev][curr] = shortest non-backtracking distance from u to curr
            # where prev is the node we came from (to avoid backtracking)
            # Use dict: (prev, curr) -> dist
            dist = {}
            # Initial: from u, we can go to any neighbor v with dist 1, prev=u
            q = deque()
            for v in adj[u]:
                dist[(u, v)] = 1
                q.append((u, v))

            while q:
                prev, curr = q.popleft()
                d = dist[(prev, curr)]

                if d >= max_depth:
                    continue

                for nxt in adj[curr]:
                    if nxt == prev:
                        continue  # non-backtracking constraint

                    # Check if we've reached the start vertex u
                    if nxt == u:
                        # Cycle length = d + 1
                        if d + 1 < g:
                            return False
                        continue

                    state = (curr, nxt)
                    if state not in dist:
                        dist[state] = d + 1
                        q.append(state)

        return True

    def build_luw_graph(q_val, d_val):
        """LUW (Lazebnik-Ustimenko-Woldar) graph construction."""
        if q_val == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q_val), (lambda x, y: (x - y) % q_val), (lambda x, y: x * y % q_val)

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}

        def get_rule(d_v):
            r = dict(rule)
            for c in range(5, d_v + 1):
                rc, c0 = (c - 5) % 4, c - (c - 5) % 4
                r[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][rc]
            return r

        r = get_rule(d_val)

        def neighbours(vertex):
            side, x = vertex
            for first in range(q_val):
                y = [None, first] + [0] * (d_val - 1)
                for c in range(2, d_val + 1):
                    a, b = r[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d_val)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    q = k

    # The girth of LUW graph with parameter d is at least 2d+2 (for appropriate rules)
    # We need girth >= g, so 2d+2 >= g => d >= (g-2)/2
    # But the actual girth depends on the specific rule. Let's try a range.
    min_d = max(3, (g - 2) // 2)

    # Try d values from min_d upward; smaller d gives smaller graphs
    for d_val in range(min_d, min_d + 20):
        if time.time() > deadline - 30:
            break
        try:
            n, edges = build_luw_graph(q, d_val)
            if n > 100000:  # Skip if too large
                continue
            if n < best_n:
                # Verify before accepting
                t_limit = 8.0 if n > 5000 else 20.0
                if verify_graph(n, edges, k, g, time_limit=t_limit):
                    best_n = n
                    best_edges = edges
                    # Since n generally increases with d, first valid is likely smallest
                    break
        except Exception:
            pass

    # Fallback: try original logic with d = g-5
    if best_edges is None and time.time() < deadline - 5:
        orig_min_d = max(3, g - 5)
        for d_val in range(orig_min_d, orig_min_d + 10):
            if time.time() > deadline - 10:
                break
            try:
                n, edges = build_luw_graph(q, d_val)
                if n <= 100000:
                    if verify_graph(n, edges, k, g, time_limit=5.0):
                        best_n = n
                        best_edges = edges
                        break
            except Exception:
                pass

    if best_edges is None:
        return 0, []

    return best_n, best_edges
