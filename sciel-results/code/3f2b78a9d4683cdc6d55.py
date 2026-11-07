def sign_edges(graph, budget):
    import numpy as np
    import time
    
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    E = len(edges)
    
    edge_u = np.array([e[0] for e in edges], dtype=np.int32)
    edge_v = np.array([e[1] for e in edges], dtype=np.int32)
    
    adj = [[] for _ in range(n)]
    for idx in range(E):
        u, v = int(edge_u[idx]), int(edge_v[idx])
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    
    def build_signed_adj(signs):
        A = np.zeros((n, n), dtype=np.float64)
        A[edge_u, edge_v] = signs.astype(np.float64)
        A[edge_v, edge_u] = signs.astype(np.float64)
        return A
    
    def spectral_radius(A):
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))
    
    def is_coboundary(signs):
        labels = np.zeros(n, dtype=np.int8)
        labels[0] = 1
        queue = [0]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v, idx in adj[u]:
                s = signs[idx]
                if labels[v] == 0:
                    labels[v] = labels[u] * s
                    queue.append(v)
                elif labels[v] != labels[u] * s:
                    return False
        return True
    
    rng = np.random.default_rng(42)
    
    best_overall_radius = np.inf
    best_overall_signs = None
    
    num_starts = 25
    time_per_start = max((deadline - start_time - 5.0) / num_starts, 0.1)
    
    for start_i in range(num_starts):
        if time.time() > deadline - 3.0:
            break
        
        signs = rng.choice([-1, 1], size=E).astype(np.int8)
        
        for _ in range(500):
            if not is_coboundary(signs):
                break
            idx = rng.integers(0, E)
            signs[idx] *= -1
        
        if is_coboundary(signs):
            continue
        
        A = build_signed_adj(signs)
        current_radius = spectral_radius(A)
        
        T = 0.2
        cooling = 0.99995
        min_T = 1e-4
        local_deadline = min(deadline - 2.0, start_time + (start_i + 1) * time_per_start)
        
        while time.time() < local_deadline:
            idx = rng.integers(0, E)
            old_sign = signs[idx]
            signs[idx] *= -1
            
            if is_coboundary(signs):
                signs[idx] = old_sign
                T *= cooling
                if T < min_T:
                    T = min_T
                continue
            
            A_new = build_signed_adj(signs)
            new_radius = spectral_radius(A_new)
            
            delta = new_radius - current_radius
            if delta < 0 or (T > 1e-10 and rng.random() < np.exp(-delta / T)):
                current_radius = new_radius
            else:
                signs[idx] = old_sign
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        if current_radius < best_overall_radius:
            best_overall_radius = current_radius
            best_overall_signs = signs.copy()
    
    if best_overall_signs is None:
        signs = np.ones(E, dtype=np.int8)
        signs[0] = -1
        return signs
    
    # Polish phase
    signs = best_overall_signs.copy()
    A = build_signed_adj(signs)
    current_radius = spectral_radius(A)
    
    T = 0.05
    cooling = 0.99999
    min_T = 1e-5
    polish_deadline = deadline - 1.0
    
    while time.time() < polish_deadline:
        idx = rng.integers(0, E)
        old_sign = signs[idx]
        signs[idx] *= -1
        
        if is_coboundary(signs):
            signs[idx] = old_sign
            T *= cooling
            if T < min_T:
                T = min_T
            continue
        
        A_new = build_signed_adj(signs)
        new_radius = spectral_radius(A_new)
        
        delta = new_radius - current_radius
        if delta < 0 or (T > 1e-10 and rng.random() < np.exp(-delta / T)):
            current_radius = new_radius
        else:
            signs[idx] = old_sign
        
        T *= cooling
        if T < min_T:
            T = min_T
    
    return signs.astype(np.int8)
