import time
import sys
from collections import deque

def construct(k, g):
    deadline = time.time() + 110.0
    
    # Helper to construct LUW graph
    def _luw_construct(q, d):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: (x * y) % q
            
        rule = {}
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            if r == 0:
                rule[c] = (1, c0 - 2)
            elif r == 1:
                rule[c] = (c0 - 1, 1)
            elif r == 2:
                rule[c] = (c0, 1)
            else:
                rule[c] = (1, c0 + 1)

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
        
        for vertex in order:
            if time.time() > deadline:
                break
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
                    
        return len(order), edges

    def _validate_graph(n, edges, k, g):
        """Validate that the graph is k-regular, simple, and has girth >= g."""
        # Check simple: no loops, no repeated edges
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                return False
            edge_set.add((u, v))
        
        # Check k-regular
        deg = [0] * n
        adj = [[] for _ in range(n)]
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u].append(v)
            adj[v].append(u)
        
        for d in deg:
            if d != k:
                return False
        
        # Check girth >= g using non-backtracking walks
        # For each vertex, do BFS tracking (node, prev) states
        # A cycle of length < g exists iff we can return to a neighbor of start
        # within < g-1 steps without backtracking
        
        for start in range(n):
            # State: (node, prev_node, depth)
            # visited: set of (node, prev_node)
            visited = set()
            queue = deque()
            for nb in adj[start]:
                queue.append((nb, start, 1))
                visited.add((nb, start))
            
            while queue:
                node, prev, depth = queue.popleft()
                if depth >= g - 1:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if nb == start:
                        # Found a cycle of length depth + 1
                        if depth + 1 < g:
                            return False
                        continue
                    state = (nb, node)
                    if state not in visited:
                        visited.add(state)
                        queue.append((nb, node, depth + 1))
        
        return True

    # For LUW construction, the girth is guaranteed to be at least 2*d + 2
    # So we need 2*d + 2 >= g, i.e., d >= (g-2)/2
    # For g=10: d >= 4
    # The number of vertices is 2 * q^d
    
    # Try the smallest d that satisfies the girth requirement
    min_d = (g - 2) // 2
    if min_d < 3:
        min_d = 3
    
    q = k
    d = min_d
    n, edges = _luw_construct(q, d)
    
    # Validate the construction
    if not _validate_graph(n, edges, k, g):
        # If validation fails, try larger d
        for d_try in range(d + 1, d + 5):
            if time.time() > deadline - 5:
                break
            n, edges = _luw_construct(q, d_try)
            if _validate_graph(n, edges, k, g):
                break
    
    return n, edges
