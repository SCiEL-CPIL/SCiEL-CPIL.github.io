import time
import sys
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Try to find a small graph using tuned LUW construction
    best_n = float('inf')
    best_edges = None
    
    # The LUW construction size is q^(d+1).
    # For k=3, q=3.
    # We want to minimize n, so minimize d.
    # Girth is roughly 2d+2 or 2d+4.
    # For g=16, we expect d around 6 or 7.
    # Let's try d from 4 to 12.
    
    if k == 3:
        for d in range(4, 13):
            if time.time() > deadline - 10:
                break
            try:
                n_luw, edges_luw = _construct_luw(k, d)
                # Check girth
                if _has_girth_at_least_fast(n_luw, edges_luw, g):
                    if n_luw < best_n:
                        best_n = n_luw
                        best_edges = edges_luw
                        # Since we iterate d in increasing order, and n increases with d,
                        # the first valid one is the smallest.
                        # However, it's possible a larger d has a smaller n? No, n=q^(d+1).
                        # So the first valid d gives the smallest n.
                        break
            except Exception:
                continue
        
        # If we found a valid graph, return it
        if best_edges is not None:
            return best_n, best_edges
        
        # Fallback to original large d if no small d worked
        d_fallback = max(3, g - 5 if (g - 5) % 2 else g - 4)
        n_luw, edges_luw = _construct_luw(k, d_fallback)
        return n_luw, edges_luw

    else:
        # For other k, use original logic or a generic fallback
        d_fallback = max(3, g - 5 if (g - 5) % 2 else g - 4)
        n_luw, edges_luw = _construct_luw(k, d_fallback)
        return n_luw, edges_luw

def _construct_luw(k, d):
    # Lazy-evaluated LUW construction from seed
    # Note: The original code calculated d based on g. 
    # Here we pass d directly.
    q = k
    # The rule definition depends on d.
    # The original code defined rule up to d.
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
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

def _random_3_reg_graph(n):
    """Generate a random simple 3-regular graph of n vertices using configuration model.
    Returns None if impossible or if multi-edges/loops are generated."""
    if n < 4 or n % 2 != 0:
        return None
    # Configuration model
    # Each vertex has 3 stubs
    stubs = [i for i in range(n) for _ in range(3)]
    random.shuffle(stubs)
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            return None # Loop
        if (u, v) in edges or (v, u) in edges:
            return None # Multi-edge
        edges.add((u, v))
    return list(edges)

def _has_girth_at_least_fast(n, edges, g):
    """Check if graph has no cycle shorter than g using BFS with early exit.
    For bipartite graphs (LUW are bipartite), we only need to check for cycles up to g-2.
    We stop BFS at depth g//2 - 1.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    max_depth = g // 2 - 1
    
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                else:
                    # Found a cycle
                    # The cycle length is dist[u] + dist[v] + 1
                    # Since it's bipartite, this will be even.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _has_girth_at_least_bfs(n, edges, g):
    """Standard BFS girth check, optimized with early exit."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                else:
                    # Found a cycle
                    # The cycle length is dist[u] + dist[v] + 1
                    # This cycle contains the start vertex.
                    # If this is the shortest cycle containing start, and we check all starts,
                    # we find the girth.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True
