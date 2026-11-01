import time
import math
import random

def construct(k, g):
    deadline = time.time() + 140.0
    
    if k == 4 and g == 10:
        return _construct_4_10_cage(deadline)
    
    return _generic_construct(k, g, deadline)

def _construct_4_10_cage(deadline):
    # Petersen graph
    base_edges = [(0,1), (1,2), (2,3), (3,4), (4,0),
                  (5,7), (7,9), (9,6), (6,8), (8,5),
                  (0,5), (1,6), (2,7), (3,8), (4,9)]
    
    n_base = 10
    group_size = 32
    n_total = n_base * group_size
    
    base_adj = [[] for _ in range(n_base)]
    for u, v in base_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)
    
    # Precompute all non-backtracking closed walks of length < g
    # For each walk, store the sequence of directed edges
    closed_walks = []
    for start in range(n_base):
        # DFS: (current, prev, path)
        stack = [(start, -1, [])]
        while stack:
            cur, prev, path = stack.pop()
            for nb in base_adj[cur]:
                if nb == prev:
                    continue
                new_path = path + [(cur, nb)]
                if nb == start and len(new_path) > 0:
                    if len(new_path) < 10:
                        closed_walks.append(new_path)
                elif len(new_path) < 10:
                    stack.append((nb, cur, new_path))
    
    random.seed(42)
    
    for attempt in range(20000):
        if time.time() > deadline - 20:
            break
        
        # Assign random voltages
        voltages = {}
        for u, v in base_edges:
            vol = random.randint(1, group_size - 1)
            voltages[(u, v)] = vol
            voltages[(v, u)] = (-vol) % group_size
        
        # Check all closed walks
        valid = True
        for walk in closed_walks:
            vol_sum = 0
            for (u, v) in walk:
                vol_sum = (vol_sum + voltages[(u, v)]) % group_size
            if vol_sum == 0:
                valid = False
                break
        
        if valid:
            edges = []
            for u, v in base_edges:
                vol = voltages[(u, v)]
                for i in range(group_size):
                    u_full = u * group_size + i
                    v_full = v * group_size + (i + vol) % group_size
                    edges.append((u_full, v_full))
            return n_total, edges
    
    # Fallback: ring construction
    n = 320
    edges = []
    for i in range(n):
        for j in [1, 2, 4, 8]:
            other = (i + j) % n
            if i < other:
                edges.append((i, other))
    return n, edges

def _generic_construct(k, g, deadline):
    n = 2 * sum((k-1)**i for i in range(g//2))
    edges = []
    for i in range(n):
        for j in range(1, k+1):
            other = (i + j) % n
            if i < other:
                edges.append((i, other))
    return n, edges
