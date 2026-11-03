import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 110.0  # 10 seconds margin
    
    # 1. Calculate Moore Bound
    # For girth g, Moore bound is 1 + k * sum_{i=0}^{g/2-1} (k-1)^i (if g even)
    # For odd g, it's slightly different, but we target even girth >= g.
    target_girth = g if g % 2 == 0 else g + 1
    
    # 2. Try LUW constructions with different d
    best_n = float('inf')
    best_edges = []
    
    # d is the dimension parameter in LUW
    # For q=5, d=5 gave ~6250.
    # We try d from 3 up to a reasonable limit (e.g., 6 or 7)
    # and check if the resulting graph has girth >= target_girth.
    
    for d in range(3, 8):
        if time.time() > deadline - 10:
            break
            
        try:
            n, edges = _construct_luw(k, g, d)
        except Exception:
            continue
            
        if n >= best_n:
            continue
            
        # Check girth
        if _check_girth(n, edges, target_girth):
            best_n = n
            best_edges = edges
            
    if best_n == float('inf'):
        # Fallback to original d calculation if no valid graph found in loop
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        best_n, best_edges = _construct_luw(k, g, d)
        
    return best_n, best_edges

def _construct_luw(k, g, d):
    q = k
    # rule generation same as seed
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q

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

def _check_girth(n, edges, target_girth):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from each vertex up to depth target_girth // 2
    for start in range(n):
        visited = {start: 0}
        parent = {start: -1}
        queue = deque([(start, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= target_girth // 2:
                continue
            for neighbor in adj[node]:
                if neighbor == parent[node]:
                    continue
                if neighbor in visited:
                    cycle_len = depth + visited[neighbor] + 1
                    if cycle_len < target_girth:
                        return False
                else:
                    visited[neighbor] = depth + 1
                    parent[neighbor] = node
                    queue.append((neighbor, depth + 1))
    return True
