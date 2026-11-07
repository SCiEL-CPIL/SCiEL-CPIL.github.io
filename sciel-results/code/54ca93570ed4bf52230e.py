import numpy as np
import time
from itertools import product

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin for 90s limit
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Build edge lookup and adjacency
    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i
        
    edge_u = np.array([u for u, v in edges], dtype=np.int32)
    edge_v = np.array([v for u, v in edges], dtype=np.int32)
    
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # Identify spanning tree and chords using BFS
    visited = [False] * n
    tree_edges = set()
    visited[0] = True
    queue = [0]
    
    while queue:
        u = queue.pop(0)
        for v, ei in adj[u]:
            if not visited[v]:
                visited[v] = True
                tree_edges.add(ei)
                queue.append(v)
    
    chord_indices = [i for i in range(num_edges) if i not in tree_edges]
    num_chords = len(chord_indices)
    
    # Precompute base matrix structure
    A_base = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        A_base[u, v] = 1.0
        A_base[v, u] = 1.0
        
    best_radius = float('inf')
    best_signs = None
    
    # Helper: check if lift is connected (has a negative cycle)
    def is_lift_connected(signs):
        parity = [-1] * n
        parity[0] = 0
        queue = [0]
        
        while queue:
            u = queue.pop(0)
            pu = parity[u]
            for v, ei in adj[u]:
                s = signs[ei]
                flip = 1 if s == -1 else 0
                expected_parity = pu ^ flip
                
                if parity[v] == -1:
                    parity[v] = expected_parity
                    queue.append(v)
                else:
                    if parity[v] != expected_parity:
                        return True
        
        return False
    
    # Helper: compute spectral radius
    def compute_radius(signs):
        A_signed = np.zeros((n, n), dtype=np.float64)
        A_signed[edge_u, edge_v] = signs.astype(np.float64)
        A_signed[edge_v, edge_u] = signs.astype(np.float64)
        eigvals = np.linalg.eigvalsh(A_signed)
        return np.max(np.abs(eigvals))
    
    # Phase 1: Enumerate all chord assignments if feasible
    if num_chords <= 14:
        for mask in range(1 << num_chords):
            if time.time() > deadline:
                break
                
            signs = np.ones(num_edges, dtype=np.int8)
            for j, idx in enumerate(chord_indices):
                if mask & (1 << j):
                    signs[idx] = -1
                    
            if not is_lift_connected(signs):
                continue
                
            radius = compute_radius(signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
    else:
        # Phase 1: Random sampling
        rng = np.random.default_rng(42)
        for _ in range(50000):
            if time.time() > deadline:
                break
                
            signs = np.ones(num_edges, dtype=np.int8)
            for idx in chord_indices:
                signs[idx] = rng.choice([-1, 1])
                
            if not is_lift_connected(signs):
                continue
                
            radius = compute_radius(signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
        
        # Phase 2: Local search from best found
        if best_signs is not None:
            current_signs = best_signs.copy()
            current_radius = best_radius
            
            # Local search: flip one edge at a time
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for ei in range(num_edges):
                    trial = current_signs.copy()
                    trial[ei] = -trial[ei]
                    
                    if not is_lift_connected(trial):
                        continue
                        
                    radius = compute_radius(trial)
                    if radius < current_radius - 1e-10:
                        current_signs = trial
                        current_radius = radius
                        best_radius = radius
                        best_signs = trial.copy()
                        improved = True
    
    # Fallback
    if best_signs is None:
        signs = np.ones(num_edges, dtype=np.int8)
        if num_chords > 0:
            signs[chord_indices[0]] = -1
        else:
            signs[0] = -1
        return signs
        
    return best_signs
