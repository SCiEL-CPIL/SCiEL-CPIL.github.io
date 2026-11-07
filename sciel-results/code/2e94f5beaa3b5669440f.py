import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    def build_signed_adj(signs):
        A = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(signs):
        A = build_signed_adj(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_connected_lift(signs):
        num_lift_vertices = 2 * n
        parent = list(range(num_lift_vertices))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            if s == 1:
                union(u * 2, v * 2)
                union(u * 2 + 1, v * 2 + 1)
            else:
                union(u * 2, v * 2 + 1)
                union(u * 2 + 1, v * 2)
        root = find(0)
        for i in range(1, num_lift_vertices):
            if find(i) != root:
                return False
        return True
    
    best_signs = None
    best_radius = float('inf')
    
    # Enumerate all 2^(m+1) switching classes
    # For each class, construct a signing with matching edges = +1
    # and check connectivity and spectral radius
    
    for pattern_idx in range(2 ** (m + 1)):
        if time.time() > deadline - 5:
            break
        
        # Decode pattern: m ladder invariants + 1 top cycle invariant
        bits = [(pattern_idx >> k) & 1 for k in range(m + 1)]
        target_ladders = [1 if b == 0 else -1 for b in bits[:m]]
        target_top = 1 if bits[m] == 0 else -1
        
        # Construct signing with matching edges = +1
        signs = np.zeros(num_edges, dtype=int)
        
        # Top edges: fix first m-1 to +1, last determined by top cycle product
        for i in range(m - 1):
            signs[i] = 1
        signs[m - 1] = target_top
        
        # Bottom edges from ladder constraints: signs[m+ip] = target_ladder_i * signs[i]
        for i in range(m):
            ip = (i + 1) % m
            signs[m + ip] = target_ladders[i] * signs[i]
        
        # Matching edges = +1
        for i in range(m):
            signs[2*m + i] = 1
        
        # Check connectivity
        if not is_connected_lift(signs):
            continue
        
        # Compute spectral radius
        r = spectral_radius(signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: try all +1
        best_signs = np.ones(num_edges, dtype=int)
    
    return best_signs
