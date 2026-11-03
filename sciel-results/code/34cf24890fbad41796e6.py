import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Strategy:
    # 1. Try LUW construction with various parameters
    # 2. Verify girth and return the smallest valid graph
    
    if k == 3 and g == 16:
        best_n = float('inf')
        best_edges = None
        
        # Try LUW constructions with different parameters
        # For k=3, q=3
        # d controls the "depth" of the construction
        # Larger d generally gives larger graphs but potentially better girth
        
        for d in [7, 8, 9]:
            if time.time() > deadline - 20:
                break
            try:
                n_luw, edges_luw = _construct_luw_optimized(k, d)
                if n_luw < best_n:
                    # Verify girth
                    if _has_girth_at_least_fast(n_luw, edges_luw, g):
                        best_n = n_luw
                        best_edges = edges_luw
            except Exception:
                continue
        
        if best_n != float('inf'):
            return best_n, best_edges
            
    # Generic fallback
    min_d = max(3, g // 2 - 2)
    max_d = g + 5
    
    for d in range(min_d, max_d + 1):
        if time.time() > deadline - 15:
            break
        try:
            n_luw, edges_luw = _construct_luw_optimized(k, d)
            if _has_girth_at_least_fast(n_luw, edges_luw, g):
                return n_luw, edges_luw
        except Exception:
            continue
                
    # Last resort - use a larger d to ensure validity
    n_luw, edges_luw = _construct_luw_optimized(k, max_d + 3)
    return n_luw, edges_luw

def _has_girth_at_least_fast(n, edges, g):
    """
    Check if graph has girth at least g using BFS from each vertex.
    Returns True if no cycle shorter than g exists.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            if dist[u] + 1 >= g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    if u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
    return True

def _construct_luw_optimized(k, d):
    """
    Optimized LUW construction for 3-regular graphs.
    Uses the standard LUW rules for k=3.
    """
    q = k  # For k=3, q=3
    
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q

    # Standard LUW rules for k=3
    # rule[c] = (a, b) where y[c] = y[a] * x[b] (or x[a] * y[b] for side 1)
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
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
