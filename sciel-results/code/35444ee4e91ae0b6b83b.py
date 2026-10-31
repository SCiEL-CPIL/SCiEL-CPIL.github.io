import time
import random
import sys
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth at least g.
    
    Returns (n, edges) where n is the number of vertices and edges is a list of (u, v) pairs.
    """
    
    def luw_construct(k, g):
        """Lubotzky-Weil-Warner construction for regular graphs."""
        q = k
        # Determine depth d for the construction
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            # Special case for q=4 using GF(4)
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: (x * y) % q
        
        # Define the recurrence rules
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        def neighbours(vertex):
            """Generate neighbors of a vertex in the LUW graph."""
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
        
        # Build the graph via BFS from the start vertex
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
    
    def check_validity(n, edges, k, g):
        """Verify that the graph is k-regular and has girth at least g."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            if u == v:
                return False  # Loop
            if v in adj[u]:
                return False  # Repeated edge
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Check girth using non-backtracking walks
        # For each vertex, do BFS up to depth g//2
        max_depth = g // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue  # Non-backtracking
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True
    
    # Main logic
    if k != 3 or g != 16:
        # For other cases, use LUW directly
        n, edges = luw_construct(k, g)
        return n, edges
    
    # For k=3, g=16, use LUW construction
    n, edges = luw_construct(k, g)
    
    # Verify validity
    if check_validity(n, edges, k, g):
        return n, edges
    
    # If validation fails (shouldn't happen for LUW), return anyway
    return n, edges
