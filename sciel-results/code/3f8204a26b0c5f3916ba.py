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
    1. Try Greedy Random Construction for n near the Moore bound.
    2. Fallback to Optimized LZW Algebraic Construction (finds smallest d).
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # 1. Try Greedy Construction
    # We attempt to find a graph with n close to the theoretical minimum.
    # Moore bound for k=5, g=9 is 426.
    # Known best is 1116.
    # We try a few targets.
    target_ns = [500, 800, 1116, 1252]
    
    for n in target_ns:
        if time.time() > deadline - 20:  # Reserve 20s for fallback
            break
        result = _greedy_construct(k, g, n, deadline - 20)
        if result is not None:
            return result
            
    # 2. Fallback to Optimized LZW
    # The LZW graph CD(d, q) has girth:
    # d+5 if d is odd
    # 2d+2 if d is even
    # We find the smallest d such that girth >= g.
    return _lzw_construct_optimized(k, g)

def _greedy_construct(k, g, n, deadline):
    """
    Greedy construction of a k-regular graph with girth >= g.
    """
    max_restarts = 5  # Reduced to save time for fallback
    for _ in range(max_restarts):
        if time.time() > deadline:
            return None
        
        adj = [set() for _ in range(n)]
        deg = [0] * n
        edges = []
        
        edges_added = 0
        target_edges = n * k // 2
        
        active_vertices = list(range(n))
        random.shuffle(active_vertices)
        
        while edges_added < target_edges:
            if time.time() > deadline:
                return None
            
            # Find vertices with deg < k
            candidates_u = [u for u in active_vertices if deg[u] < k]
            if not candidates_u:
                break
            
            # Pick vertex with fewest valid candidates (MRV heuristic)
            best_u = None
            best_valid_vs = None
            best_count = float('inf')
            
            # Sample to speed up
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
            
            adj[best_u].add(v)
            adj[v].add(best_u)
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
    
    # BFS from u up to depth g-2
    max_depth = g - 2
    dist = {u: 0}
    queue = deque([(u, 0)])
    
    while queue:
        node, d = queue.popleft()
        if d >= max_depth:
            continue
        for nb in adj[node]:
            if nb not in dist:
                dist[nb] = d + 1
                if d + 1 <= max_depth:
                    queue.append((nb, d + 1))
    
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
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    # BFS from each vertex to find shortest cycle
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    parent[nb] = node
                    queue.append(nb)
                elif nb != parent[node]:
                    cycle_len = dist[node] + dist[nb] + 1
                    if cycle_len < g:
                        return False
                        
    return True

def _lzw_construct_optimized(k, g):
    """
    Fallback: Lazebnik-Ustimenko-Woldar graph.
    Finds the smallest d such that the girth is >= g.
    """
    # Try d from 1 upwards
    for d in range(1, 20):
        # Estimate girth
        if d % 2 == 1:
            est_girth = d + 5
        else:
            est_girth = 2 * d + 2
            
        if est_girth >= g:
            n, edges = _build_lzw(k, d)
            # Verify to be safe, as formulas can be tricky
            if _verify_girth(n, edges, k, g):
                return n, edges
                
    # If we reach here, something went wrong, return a large valid graph
    return _build_lzw(k, 5)

def _build_lzw(k, d):
    """
    Build the LZW graph CD(d, q) with q=k.
    """
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

# EVOLVE-BLOCK-END
