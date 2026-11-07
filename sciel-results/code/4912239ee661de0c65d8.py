import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Precompute edge structure for fast sign updates
    edge_u = np.array([e[0] for e in edges], dtype=np.int32)
    edge_v = np.array([e[1] for e in edges], dtype=np.int32)
    
    def build_signed_adj(signs):
        A = np.zeros((n, n))
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        return A
    
    def spectral_radius(signs):
        A = build_signed_adj(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def trace_A4(signs):
        """Fast proxy: trace(A^4) = 4*sum(d_i^2) + 2*sum over 4-cycles... 
        For signed graph, trace(A^4) = 4*sum_v d(v)^2 + 2*sum over closed walks of length 4.
        Actually simpler: just compute A^2 then trace(A^2 @ A^2)."""
        A = build_signed_adj(signs)
        A2 = A @ A
        return np.trace(A2 @ A2)
    
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
        for idx in range(num_edges):
            u = edge_u[idx]
            v = edge_v[idx]
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
    
    def local_search(signs, max_iterations=100, use_proxy=True):
        """Try flipping individual signs to reduce spectral radius (or trace_A4 proxy)."""
        current_signs = signs.copy()
        if use_proxy:
            current_metric = trace_A4(current_signs)
        else:
            current_metric = spectral_radius(current_signs)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            improved = False
            for idx in range(num_edges):
                trial_signs = current_signs.copy()
                trial_signs[idx] *= -1
                if not is_connected_lift(trial_signs):
                    continue
                if use_proxy:
                    trial_metric = trace_A4(trial_signs)
                else:
                    trial_metric = spectral_radius(trial_signs)
                if trial_metric < current_metric - 1e-10:
                    current_signs = trial_signs
                    current_metric = trial_metric
                    improved = True
                    break
            if not improved:
                break
        
        return current_signs
    
    def make_signing(target_ladders, target_top):
        """Construct signing with matching edges = +1 given ladder and top cycle products."""
        signs = np.zeros(num_edges, dtype=int)
        for i in range(m - 1):
            signs[i] = 1
        signs[m - 1] = target_top
        for i in range(m):
            ip = (i + 1) % m
            signs[m + ip] = target_ladders[i] * signs[i]
        for i in range(m):
            signs[2*m + i] = 1
        return signs
    
    best_signs = None
    best_radius = float('inf')
    
    # Enumerate all 2^(m+1) switching classes
    for pattern_idx in range(2 ** (m + 1)):
        if time.time() > deadline - 10:
            break
        
        bits = [(pattern_idx >> k) & 1 for k in range(m + 1)]
        target_ladders = [1 if b == 0 else -1 for b in bits[:m]]
        target_top = 1 if bits[m] == 0 else -1
        
        signs = make_signing(target_ladders, target_top)
        
        if not is_connected_lift(signs):
            continue
        
        r = spectral_radius(signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()
    
    # Local search from best found so far using trace_A4 proxy (fast)
    if best_signs is not None and time.time() < deadline - 20:
        polished_signs = local_search(best_signs, max_iterations=100, use_proxy=True)
        polished_radius = spectral_radius(polished_signs)
        if polished_radius < best_radius:
            best_signs = polished_signs
            best_radius = polished_radius
    
    # High-precision polish: use actual spectral radius for fine-tuning
    if best_signs is not None and time.time() < deadline - 8:
        polished_signs = local_search(best_signs, max_iterations=50, use_proxy=False)
        polished_radius = spectral_radius(polished_signs)
        if polished_radius < best_radius - 1e-12:
            best_signs = polished_signs
            best_radius = polished_radius
    
    # Try a few more starting points if time permits
    if time.time() < deadline - 25:
        all_plus = np.ones(num_edges, dtype=int)
        if is_connected_lift(all_plus):
            polished_signs = local_search(all_plus, max_iterations=50, use_proxy=True)
            polished_radius = spectral_radius(polished_signs)
            if polished_radius < best_radius:
                best_signs = polished_signs
                best_radius = polished_radius
    
    if time.time() < deadline - 20:
        # Try alternating pattern
        alt = np.ones(num_edges, dtype=int)
        for i in range(0, num_edges, 2):
            alt[i] = -1
        if is_connected_lift(alt):
            polished_signs = local_search(alt, max_iterations=50, use_proxy=True)
            polished_radius = spectral_radius(polished_signs)
            if polished_radius < best_radius:
                best_signs = polished_signs
                best_radius = polished_radius
    
    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=int)
    
    return best_signs
