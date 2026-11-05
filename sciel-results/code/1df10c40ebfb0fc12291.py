"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from collections import deque

# EVOLVE-BLOCK-START

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    # Ensure d is large enough for girth g
    # Girth of CD(d, q) is at least d + 5 for odd d? 
    # Actually, the girth is related to d.
    # For (5,9), we need girth 9.
    # If we use d=4, girth might be 9?
    # Let's stick to the seed's logic: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
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

def _check_girth(adj, g):
    """
    Check if graph has girth at least g using BFS from each vertex.
    Optimized for small g.
    """
    n = len(adj)
    if n == 0:
        return True
    
    # We only need to check for cycles of length 3 to g-1
    # If g <= 3, any edge is a cycle of length 2? No, simple graph.
    # Girth is length of shortest cycle.
    # If g=9, we check for cycles of length 3,4,5,6,7,8.
    
    # To speed up, we can use a global visited array with timestamps?
    # Or just standard BFS.
    
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            # If dist[u] is already large, we might not find shorter cycles
            # but we still need to check edges from u.
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # If v is visited and not the parent of u, we found a cycle
                    # Cycle length is dist[u] + dist[v] + 1
                    if parent[u] != v:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
    return True

def _try_lift_k6(m, deadline):
    """
    Try to construct a voltage graph lift of K6 with size m.
    Returns (n, edges) if a valid graph with girth >= 9 is found, else None.
    """
    # K6 vertices: 0..5
    # Edges: all pairs (i, j) with i < j
    base_edges = []
    for i in range(6):
        for j in range(i+1, 6):
            base_edges.append((i, j))
            
    # Number of edges is 15
    # Assign random voltages
    # We will try a few random assignments
    for _ in range(10):
        voltages = {}
        for u, v in base_edges:
            voltages[(u, v)] = random.randint(0, m - 1)
            # Symmetric voltage for undirected graph?
            # In lift, edge (u, a) to (v, a + vol).
            # The reverse edge (v, a') to (u, a' - vol).
            # So we just need vol for (u, v).
            
        # Construct lift
        n = 6 * m
        adj = [[] for _ in range(n)]
        edges = []
        
        for u, v in base_edges:
            vol = voltages[(u, v)]
            for a in range(m):
                u_lift = u * m + a
                v_lift = v * m + (a + vol) % m
                adj[u_lift].append(v_lift)
                adj[v_lift].append(u_lift)
                if u_lift < v_lift:
                    edges.append((u_lift, v_lift))
                else:
                    edges.append((v_lift, u_lift))
                    
        # Check girth
        # To save time, we can do a quick check
        if _check_girth(adj, 9):
            return n, edges
            
        if time.time() > deadline:
            break
            
    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    best_n = float('inf')
    best_edges = None
    
    # Try voltage graph lifts of K6 for (5, 9)
    if k == 5 and g == 9:
        # Try various m values
        # Moore bound is ~1700. 6*m = 1700 => m ~ 283.
        # Let's try m from 100 to 300.
        for m in range(100, 350, 10):
            if time.time() > deadline:
                break
            
            # Try a few random assignments for this m
            result = _try_lift_k6(m, deadline)
            if result:
                n, edges = result
                if n < best_n:
                    best_n = n
                    best_edges = edges
                    
            # If we found a very good one (close to Moore bound), we can stop early
            if best_n < 1800:
                break
                
    # If we found a valid lift, return it
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback to LUW construction
    n, edges = _luw_construct(k, g)
    return n, edges

# EVOLVE-BLOCK-END
