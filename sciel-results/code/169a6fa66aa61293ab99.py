import time
import sys
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    if k != 3:
        # Fallback for other k
        d_fallback = max(3, g - 5 if (g - 5) % 2 else g - 4)
        n_luw, edges_luw = _construct_luw(k, d_fallback)
        return n_luw, edges_luw
    
    # Strategy: Try voltage lifting of the (3,10)-cage
    best_n = float('inf')
    best_edges = None
    
    # Get the (3,10)-cage (70 vertices, girth 10)
    base_n, base_edges = _get_3_10_cage()
    
    if base_n == 0:
        # Fallback to LUW
        for d in range(4, 13):
            if time.time() > deadline - 10:
                break
            try:
                n_luw, edges_luw = _construct_luw(k, d)
                if _has_girth_at_least_fast(n_luw, edges_luw, g):
                    if n_luw < best_n:
                        best_n = n_luw
                        best_edges = edges_luw
                    break
            except Exception:
                continue
        if best_edges is not None:
            return best_n, best_edges
        d_fallback = max(3, g - 5 if (g - 5) % 2 else g - 4)
        n_luw, edges_luw = _construct_luw(k, d_fallback)
        return n_luw, edges_luw
    
    # Try different lift factors
    # n_lifted = base_n * L
    # We want n_lifted around 900-936
    # base_n = 70, so L should be around 13-14
    for L in range(12, 16):
        if time.time() > deadline - 15:
            break
        
        n_lifted = base_n * L
        if n_lifted >= 936:
            continue
        
        # Try multiple random voltage assignments
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            # Assign random permutations from S_3 to each edge
            # Permutations: 0=identity, 1=(0 1 2), 2=(0 2 1)
            voltages = {}
            for u, v in base_edges:
                voltages[(u, v)] = random.randint(0, 2)
                voltages[(v, u)] = (3 - voltages[(u, v)]) % 3  # Inverse
        
        # Build the lifted graph
        lifted_edges = _build_lifted_graph(base_n, base_edges, L, voltages)
        
        # Check girth
        if _has_girth_at_least_fast(n_lifted, lifted_edges, g):
            if n_lifted < best_n:
                best_n = n_lifted
                best_edges = lifted_edges
                break
        
        if best_edges is not None and best_n <= 910:
            break
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback to LUW
    for d in range(4, 13):
        if time.time() > deadline - 5:
            break
        try:
            n_luw, edges_luw = _construct_luw(k, d)
            if _has_girth_at_least_fast(n_luw, edges_luw, g):
                return n_luw, edges_luw
        except Exception:
            continue
    
    d_fallback = max(3, g - 5 if (g - 5) % 2 else g - 4)
    n_luw, edges_luw = _construct_luw(k, d_fallback)
    return n_luw, edges_luw

def _get_3_10_cage():
    """Return the (3,10)-cage: 70 vertices, 3-regular, girth 10.
    This is the Biggs-Smith graph? No, that's (3,9).
    The (3,10)-cage is a known graph with 70 vertices.
    I'll use a known construction or hardcoded edges.
    """
    # The (3,10)-cage can be constructed from the incidence graph of a generalized quadrangle
    # or from known tables. For now, I'll use a deterministic construction.
    
    # Actually, let me use the LUW construction with d=4 which gives n=3^5=243.
    # But that's too big. Let me try a different approach.
    
    # For the (3,10)-cage, I'll use a known 70-vertex graph.
    # Let me construct it using a different method or hardcode it.
    
    # Since I don't have the exact edges hardcoded, let me use LUW with a smaller d
    # and check if it has girth 10.
    
    # LUW with d=3: n = 3^4 = 81
    # LUW with d=4: n = 3^5 = 243
    
    # Let me try d=3 and see the girth
    try:
        n, edges = _construct_luw(3, 3)
        if _has_girth_at_least_fast(n, edges, 10):
            return n, edges
    except Exception:
        pass
    
    # Try d=4
    try:
        n, edges = _construct_luw(3, 4)
        if _has_girth_at_least_fast(n, edges, 10):
            return n, edges
    except Exception:
        pass
    
    # If LUW doesn't work, return 0 to signal fallback
    return 0, []

def _build_lifted_graph(base_n, base_edges, L, voltages):
    """Build the L-fold voltage lift of the base graph.
    
    Vertices in the lift: (v, s) where v in 0..base_n-1, s in 0..L-1
    Edge (u, s) to (v, t) where t = s + voltage(u,v) mod L
    """
    lifted_edges = []
    
    for u, v in base_edges:
        vol = voltages.get((u, v), 0)
        for s in range(L):
            t = (s + vol) % L
            # Vertex (u, s) has index u * L + s
            # Vertex (v, t) has index v * L + t
            lifted_edges.append((u * L + s, v * L + t))
    
    return lifted_edges

def _construct_luw(k, d):
    """LUW construction."""
    q = k
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

def _has_girth_at_least_fast(n, edges, g):
    """Check if graph has no cycle shorter than g using BFS with early exit."""
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
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True
