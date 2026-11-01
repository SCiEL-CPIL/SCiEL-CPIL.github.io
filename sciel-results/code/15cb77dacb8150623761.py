import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145

    # Petersen graph edges
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    petersen_n = 10

    # Heawood graph edges
    heawood_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
        (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 6),
        (0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11)
    ]
    heawood_n = 14

    bases = [
        (petersen_n, petersen_edges),
        (heawood_n, heawood_edges)
    ]

    candidates = []
    max_n = 1000
    
    for base_n, base_edges in bases:
        m_min = (512 + base_n - 1) // base_n
        m_max = max_n // base_n
        for m in range(m_min, m_max + 1):
            candidates.append((base_n * m, base_n, base_edges, m))
            
    candidates.sort(key=lambda x: x[0])
    
    for n, base_n, base_edges, m in candidates:
        if time.time() > deadline:
            break
            
        attempts = 10
        for _ in range(attempts):
            if time.time() > deadline:
                break
            shifts = [random.randint(0, m-1) for _ in range(len(base_edges))]
            edges = _build_lift(base_edges, m, shifts, base_n)
            if _verify_girth_fast(n, edges, g):
                return n, edges
                
    # Fallback
    m = 100
    shifts = [random.randint(0, m-1) for _ in range(len(petersen_edges))]
    n = petersen_n * m
    edges = _build_lift(petersen_edges, m, shifts, petersen_n)
    return n, edges

def _build_lift(base_edges, m, shifts, base_n):
    n = base_n * m
    edges = []
    for (u, v), shift in zip(base_edges, shifts):
        for i in range(m):
            u_id = u * m + i
            v_id = v * m + (i + shift) % m
            edges.append((u_id, v_id))
    return edges

def _verify_girth_fast(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                elif v != u:
                    # Check if it's a back edge that forms a short cycle
                    # We must ensure v is not the parent of u.
                    # In this simple BFS, we don't track parent.
                    # But if dist[v] == dist[u] + 1, it's a tree edge (or cross edge in same level? No, BFS levels).
                    # If dist[v] == dist[u] - 1, it's the parent (usually).
                    # If dist[v] == dist[u], it's a cross edge.
                    # If dist[v] < dist[u] - 1, it's a back edge.
                    # Actually, any edge (u,v) where dist[v] != -1 and (u,v) is not a tree edge forms a cycle.
                    # The cycle length is dist[u] + dist[v] + 1.
                    # We just need to check if this is < g.
                    # To avoid double counting or checking tree edges, we can just check all visited neighbors.
                    # If dist[u] + dist[v] + 1 < g, return False.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

# EVOLVE-BLOCK-END
