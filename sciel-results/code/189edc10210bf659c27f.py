import time
import random
import sys

# EVOLVE-BLOCK-START

def _verify_girth_bfs(adj, n, g):
    """Verify girth >= g using BFS from each vertex, tracking parent to avoid backtracking."""
    for start in range(n):
        # BFS: dist[v] = distance from start, parent[v] = parent in BFS tree
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[v] <= dist[u]:
                    # Found a cycle of length dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _bfs_component(adj, n, start):
    """Return the set of vertices in the connected component of start."""
    visited = set()
    queue = [start]
    visited.add(start)
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                queue.append(v)
    return visited

def _construct_luw(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q). Returns (n, edges) or None if invalid."""
    q = k
    # d is chosen so that girth >= d + 5 (for odd d) or d + 4 (for even d)
    # We want girth >= g
    # For odd d: girth >= d + 5, so d >= g - 5
    # For even d: girth >= d + 4, so d >= g - 4
    # We pick d = g - 5 if (g-5) is odd, else g - 4
    d = g - 5
    if d % 2 == 0:
        d = g - 4
    if d < 3:
        d = 3
    
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
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # Build the graph via BFS from the zero point
    # Vertices are (side, tuple of length d+1) where side in {0, 1}
    # side 0 = points, side 1 = lines
    # y[0] is None for points, y[1] = first coordinate
    # Actually from the original: side, x where x is a tuple
    # For points: x = (None, c1, c2, ..., cd) where ci are in GF(q)
    # For lines: similar
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point p ~ line l iff l[c] - p[c] = l[a] * p[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {}
    order = []
    edges = []
    
    index[start] = 0
    order.append(start)
    
    for vi, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    # Verify
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return None
    # Check no multi-edges
    for i in range(n):
        if len(adj[i]) != len(set(adj[i])):
            return None
    
    # Check girth
    if not _verify_girth_bfs(adj, n, g):
        return None
    
    return n, edges

def construct(k, g):
    deadline = time.time() + 140.0  # 10s safety margin
    
    # Try LUW construction
    result = _construct_luw(k, g)
    if result is not None:
        n, edges = result
        return n, edges
    
    # If LUW fails, fall back to a simple construction
    # For small cases, we can try a different approach
    # But for the given task (k=5, g=9), LUW should work
    
    # Fallback: raise an error if we can't construct
    raise ValueError("Could not construct a valid graph")

# EVOLVE-BLOCK-END
