"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    Strategy:
    1. Try Greedy Random Construction for increasing n.
    2. Fallback to LZW Algebraic Construction.
    """
    deadline = time.time() + 110  # Safety margin
    
    # Try greedy construction for smaller n first
    # The Moore bound for k=5, g=9 is 426.
    # Best known is 1116.
    # We try to find something smaller than the LZW fallback (1562).
    target_ns = [1100, 1200, 1300, 1400, 1500]
    
    for n in target_ns:
        if time.time() > deadline:
            break
        result = _greedy_construct(k, g, n, deadline)
        if result is not None:
            return result
            
    # Fallback to LZW
    return _lzw_construct(k, g)

def _greedy_construct(k, g, n, deadline):
    """
    Greedy construction of a k-regular graph with girth >= g.
    """
    # We will try a few times for each n
    max_restarts = 5
    for _ in range(max_restarts):
        if time.time() > deadline:
            return None
        
        adj = [[] for _ in range(n)]
        deg = [0] * n
        edges = []
        
        edges_added = 0
        target_edges = n * k // 2
        
        active_vertices = list(range(n))
        random.shuffle(active_vertices)
        
        while edges_added < target_edges:
            if time.time() > deadline:
                return None
            
            candidates_u = [u for u in active_vertices if deg[u] < k]
            if not candidates_u:
                break
            
            # Pick vertex with fewest valid candidates to avoid getting stuck
            best_u = None
            best_valid_vs = None
            best_count = float('inf')
            
            # Sample a subset of candidates_u to save time
            sample_size = min(len(candidates_u), 20)
            sampled = random.sample(candidates_u, sample_size)
            
            for u in sampled:
                valid_vs = _find_valid_neighbors(adj, deg, u, k, g, n)
                if len(valid_vs) < best_count:
                    best_count = len(valid_vs)
                    best_u = u
                    best_valid_vs = valid_vs
                    if best_count == 0:
                        break
            
            if best_u is None or len(best_valid_vs) == 0:
                return None
            
            v = random.choice(best_valid_vs)
            
            adj[best_u].append(v)
            adj[v].append(best_u)
            deg[best_u] += 1
            deg[v] += 1
            edges.append((best_u, v))
            edges_added += 1
            
        if all(d == k for d in deg):
            if _verify_girth(n, edges, k, g):
                return (n, edges)
                
    return None

def _find_valid_neighbors(adj, deg, u, k, g, n):
    """Find all valid neighbors v for u that maintain girth constraint."""
    forbidden = set(adj[u])
    forbidden.add(u)
    
    # BFS from u up to depth g-2 to find all vertices within distance g-2
    # These vertices cannot be connected to u
    dist = {u: 0}
    queue = deque([(u, 0)])
    max_depth = g - 2
    
    while queue:
        node, d = queue.popleft()
        if d >= max_depth:
            continue
        for nb in adj[node]:
            if nb not in dist:
                dist[nb] = d + 1
                if d + 1 <= max_depth:
                    queue.append((nb, d + 1))
    
    # All vertices in dist are within distance <= g-2 from u
    # So they cannot be neighbors of u
    # Valid v: v not in dist, v not in forbidden, deg[v] < k
    valid_vs = []
    for v in range(n):
        if v in dist or v in forbidden:
            continue
        if deg[v] < k:
            valid_vs.append(v)
    
    return valid_vs

def _verify_girth(n, edges, k, g):
    """
    Verify the graph has girth >= g and is k-regular.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    # BFS from each vertex to find shortest cycle
    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        
        for node in queue:
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    parent[nb] = node
                    queue.append(nb)
                elif nb != parent[node]:
                    cycle_len = dist[node] + dist[nb] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth < g:
                            return False
                            
    return min_girth >= g

def _lzw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar graph"""
    q = k
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

# EVOLVE-BLOCK-END
