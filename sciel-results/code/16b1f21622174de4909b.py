import time
import sys
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START

def _get_heawood_edges():
    """
    Returns the list of edges for the Heawood graph (14 vertices, 3-regular, girth 6).
    Vertices 0-6 are points, 7-13 are lines of the Fano plane.
    """
    # Correct Fano plane lines:
    lines = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    edges = []
    for i, line in enumerate(lines):
        line_vertex = 7 + i
        for point in line:
            edges.append((point, line_vertex))
            
    return edges

def _build_lift(base_edges, m, shifts):
    n = 14 * m
    edges = []
    for (u, v), shift in zip(base_edges, shifts):
        for i in range(m):
            u_id = u * m + i
            v_id = v * m + (i + shift) % m
            edges.append((u_id, v_id))
    return n, edges

def _verify_girth_fast(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    dist = np.full(n, -1, dtype=np.int32)
    parent = np.full(n, -1, dtype=np.int32)
    seen_mark = np.zeros(n, dtype=np.int32)
    mark = 0
    half_g = g // 2

    for s in range(n):
        mark += 1
        dist[s] = 0
        parent[s] = -1
        seen_mark[s] = mark
        queue = deque([s])

        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= half_g:
                continue
            for v in adj[u]:
                if seen_mark[v] != mark:
                    dist[v] = du + 1
                    parent[v] = u
                    seen_mark[v] = mark
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False

    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145

    base_edges = _get_heawood_edges()
    num_base_edges = len(base_edges)
    
    # Heawood has 21 edges
    m = 20

    while m <= 1000 and time.time() < deadline:
        # Increase trials for small m
        num_trials = 2000 if m < 50 else 500
        
        for _ in range(num_trials):
            if time.time() > deadline:
                break
            shifts_rand = [random.randint(0, m-1) for _ in range(num_base_edges)]
            n, edges = _build_lift(base_edges, m, shifts_rand)
            if _verify_girth_fast(n, edges, g):
                return n, edges
        
        m += 1

    # Fallback
    m = 3000
    shifts_rand = [random.randint(0, m-1) for _ in range(num_base_edges)]
    n, edges = _build_lift(base_edges, m, shifts_rand)
    return n, edges

# EVOLVE-BLOCK-END
