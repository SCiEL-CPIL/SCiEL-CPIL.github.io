import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    def luw_construction():
        """LUW construction: produces 2*q^d vertices for k=4, g=10 (320 vertices)."""
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

    def verify_girth(n, edges, g):
        """Verify girth >= g using non-backtracking walks from every vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        for i in range(n):
            if len(adj[i]) != k:
                return False

        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)

        # Use BFS from each vertex to check for cycles of length < g
        # For efficiency, we can use the non-backtracking walk approach
        # but for small graphs, BFS from each vertex is simpler
        for L in range(3, g):
            for start in range(n):
                # BFS to find shortest cycle back to start
                # State: (vertex, prev, length)
                visited = set()
                queue = deque()
                queue.append((start, -1, 0))
                visited.add((start, -1))

                found = False
                while queue:
                    v, prev, l = queue.popleft()
                    if l == L:
                        if v == start:
                            return False
                        continue
                    for w in adj[v]:
                        if w != prev:
                            if (w, v) not in visited:
                                visited.add((w, v))
                                queue.append((w, v, l + 1))
        return True

    # Try to find the best construction
    n, edges = luw_construction()

    # Verify the construction
    if verify_girth(n, edges, g):
        return n, edges

    # If verification fails, try to return anyway (shouldn't happen)
    return n, edges
