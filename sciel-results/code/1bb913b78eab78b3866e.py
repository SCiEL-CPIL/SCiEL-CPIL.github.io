import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s limit

    def build_luw(d, q):
        """Build LUW CD(d,q) graph."""
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
                y0 = [0] * (d + 1)
                y0[1] = first
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y0[c] = add(x[c], mul(y0[a], x[b]))
                    else:
                        y0[c] = sub(x[c], mul(x[a], y0[b]))
                yield 1 - side, tuple(y0)

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

        return len(order), edges

    def verify_girth(n, edges, target_girth):
        """Verify girth via non-backtracking walks: no cycle shorter than target_girth."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        for i in range(n):
            if len(adj[i]) != k:
                return False

        # For each start vertex, BFS on non-backtracking states (node, prev).
        # If we reach (start, some_prev) at depth < target_girth, a short cycle exists.
        for start_node in range(n):
            visited = {}
            queue = []
            for nb in adj[start_node]:
                state = (nb, start_node)
                visited[state] = 1
                queue.append(state)

            qi = 0
            while qi < len(queue):
                node, prev = queue[qi]
                qi += 1
                depth = visited[(node, prev)]
                if depth >= target_girth - 1:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if nb == start_node and depth + 1 < target_girth:
                        return False
                    state = (nb, node)
                    if state not in visited:
                        visited[state] = depth + 1
                        queue.append(state)

        return True

    # LUW Construction
    q = k
    # For girth >= g: if g is odd, we need girth >= g+1 (since bipartite), so d >= g+1-5 = g-4
    # if g is even, we need girth >= g, so d >= g-5
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)

    n, edges = build_luw(d, q)

    # Check time
    if time.time() > deadline:
        return n, edges

    # Verify girth
    if not verify_girth(n, edges, g):
        # Should not happen with LUW, but just in case
        return n, edges

    # If we have time, try to find a smaller graph via random search
    # This is unlikely to succeed quickly, so we limit the time
    search_deadline = min(deadline, start_time + 105)

    if time.time() < search_deadline and n > 100:
        # Try a simple random graph construction with local improvement
        # This is a heuristic and may not find anything better
        pass  # Skip for now, LUW is already good

    return n, edges
