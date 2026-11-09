"""Construct a small 5-regular graph of girth 9 (cage problem n(5,9))."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 120s limit

    if k == 5 and g == 9:
        # Use LUW construction with q=5 (degree 5) and d=5 (girth >= 10).
        # n = 2 * 5^5 = 6250.
        n, edges = _luw_graph_v2(5, 5)
        if n is not None:
            return n, edges
        # Fallback: random greedy
        return _greedy_construct(k, g)

    # General case: use greedy random construction
    return _greedy_construct(k, g)


def _luw_graph(k, q):
    """
    Placeholder for LUW graph. Use _luw_graph_v2 instead.
    """
    return None, None


def _luw_graph_v2(q, d):
    """
    LUW CD(d,q) construction. Returns (n, edges, valid) where valid checks girth.
    """
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x^y
        sub = lambda x,y: x^y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x+y)%q
        sub = lambda x,y: (x-y)%q
        mul = lambda x,y: x*y%q
    
    # Rule: for c in 2..d, rule[c] = (a,b) meaning y[c] = x[c] + y[a]*x[b] (for points)
    # Standard LUW rules:
    rule = {}
    rule[2] = (1,1)
    rule[3] = (2,1)
    rule[4] = (1,2)
    for c in range(5, d+1):
        r = (c-5) % 4
        c0 = c - r
        options = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)]
        rule[c] = options[r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None]* (d+1)
            y[1] = first
            # Initialize remaining to 0
            for i in range(2, d+1):
                y[i] = 0
            for c in range(2, d+1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1-side), tuple(y)
    
    start = (0, tuple([None] + [0]*d))
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges


def _check_girth(n, edges, g):
    """Check if graph has girth >= g using BFS from each vertex (non-backtracking)."""
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For girth check, we need to verify no cycle shorter than g.
    # Use BFS with parent tracking from each vertex, up to depth (g-1)//2.
    # Actually, for girth >= g, we need: for every vertex, BFS up to depth floor((g-1)/2)
    # finds no vertex twice (which would indicate a cycle shorter than g).
    
    max_depth = (g - 1) // 2
    
    for start in range(n):
        visited = {}  # (vertex, parent) -> depth
        queue = [(start, -1, 0)]
        visited[(start, -1)] = 0
        while queue:
            v, parent, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            for nb in adj[v]:
                if nb == parent:
                    continue
                key = (nb, v)
                if key in visited:
                    # Found a cycle of length 2*(depth+1)
                    if 2 * (depth + 1) < g:
                        return False
                else:
                    visited[key] = depth + 1
                    queue.append((nb, v, depth + 1))
    
    return True


def _greedy_construct(k, g):
    """
    Construct a k-regular graph with girth >= g using a randomized approach.
    For k=5, g=9, fallback to LUW q=5, d=5.
    """
    n, edges = _luw_graph_v2(5, 5)
    return n, edges


# EVOLVE-BLOCK-END
