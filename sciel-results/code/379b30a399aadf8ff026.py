import numpy as np
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction.
    For k=5, tries d=4 first (1250 vertices), falls back to d=5.
    """
    q = k  # q = 5 for k=5

    def get_luw_graph(d, q):
        """Constructs the LUW graph CD(d, q)."""
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        # Define the rule for LUW graph based on Lazebnik, Ustimenko, Woldar (1998)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
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

        return len(order), edges

    def check_girth(n, edges, target_girth):
        """
        Returns True if girth is >= target_girth.
        Uses non-backtracking walk verification from every vertex.
        """
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Verify every vertex has degree exactly k
        for i in range(n):
            if len(adj[i]) != k:
                return False

        # Check for duplicate edges
        edge_set = set()
        for u, v in edges:
            if u >= v:
                continue
            if (u, v) in edge_set:
                return False
            edge_set.add((u, v))

        # Non-backtracking BFS to find shortest cycle
        # For each starting vertex s, do BFS but don't go back to parent
        # Track (vertex, parent) pairs; a cycle is found if we revisit
        # a (vertex, parent) pair.
        
        for s in range(n):
            # State: (vertex, parent) where parent is the vertex we came from
            # Distance from s
            dist = {}
            queue = deque()
            for nb in adj[s]:
                dist[(nb, s)] = 1
                queue.append((nb, s))
            
            while queue:
                u, p = queue.popleft()
                d = dist[(u, p)]
                if d >= target_girth:
                    break
                for v in adj[u]:
                    if v == p:
                        continue  # non-backtracking
                    if v == s:
                        # Found a cycle of length d + 1
                        if d + 1 < target_girth:
                            return False
                        break
                    state = (v, u)
                    if state not in dist:
                        dist[state] = d + 1
                        queue.append(state)
        
        return True

    # Try d=4 first (1250 vertices), then d=5 (6250 vertices)
    for d in [4, 5]:
        n, edges = get_luw_graph(d, q)
        if check_girth(n, edges, g):
            return n, edges

    # Should not reach here for k=5, g=9
    raise RuntimeError("Failed to construct valid graph")
