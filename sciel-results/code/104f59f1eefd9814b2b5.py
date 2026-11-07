import time
import numpy as np
from collections import deque

def sign_edges(graph, budget):
    """
    Returns an array of signs (-1 or +1) for each edge in the graph.
    Strategy:
    1. Precompute adjacency list for fast connectivity checks.
    2. Use fast incremental spectral radius updates.
    3. Exhaustive search for small m (m=4, m=5), then multi-restart local search
       with single-flip and double-flip moves, plus a final high-precision polish.
    4. Add simulated annealing to escape local minima.
    """
    
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Precompute adjacency list once
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    u_arr = np.array([e[0] for e in edges])
    v_arr = np.array([e[1] for e in edges])
    
    def is_lift_connected(signs):
        """
        Returns True if the 2-lift is connected (signing is NOT a coboundary).
        """
        f = [-1] * n
        f[0] = 0
        queue = deque([0])
        
        while queue:
            u = queue.popleft()
            for v, idx in adj[u]:
                s = signs[idx]
                flip = 1 if s == -1 else 0
                required_f_v = f[u] ^ flip
                
                if f[v] == -1:
                    f[v] = required_f_v
                    queue.append(v)
                else:
                    if f[v] != required_f_v:
                        return True
        
        return False

    def build_matrix(signs):
        S = np.zeros((n, n), dtype=np.float64)
        S[u_arr, v_arr] = signs.astype(np.float64)
        S[v_arr, u_arr] = signs.astype(np.float64)
        return S

    def get_radius(signs):
        S = build_matrix(signs)
        eigvals = np.linalg.eigvalsh(S)
        return float(np.max(np.abs(eigvals)))

    def fast_radius_update(S, idx, old_sign, new_sign):
        """
        Fast spectral radius computation after flipping edge idx.
        """
        S_new = S.copy()
        u, v = edges[idx]
        delta = new_sign - old_sign
        S_new[u, v] += delta
        S_new[v, u] += delta
        eigvals = np.linalg.eigvalsh(S_new)
        return float(np.max(np.abs(eigvals)))

    best_signs = None
    best_radius = float('inf')
    
    rng = np.random.default_rng(3)
    
    # Strategy 1: Exhaustive Search for small m
    if num_edges <= 15:
        # m=4: 12 edges, 2^12 = 4096
        # m=5: 15 edges, 2^15 = 32768
        max_combo = 1 << num_edges
        for k in range(max_combo):
            if time.time() > deadline - 2.0:
                break
            signs = np.where((k >> np.arange(num_edges)) & 1, 1, -1).astype(np.int8)
            if not is_lift_connected(signs):
                continue
            S = build_matrix(signs)
            eigvals = np.linalg.eigvalsh(S)
            radius = float(np.max(np.abs(eigvals)))
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
        if best_signs is not None:
            return best_signs

    # Strategy 2: Multi-restart local search with single and double flips
    
    def local_search(signs, max_time=5.0):
        """
        Local search with single-flip and occasional double-flip moves.
        Returns (best_signs, best_radius) found.
        """
        if not is_lift_connected(signs):
            return None, float('inf')
        
        current_S = build_matrix(signs)
        current_radius = float(np.max(np.abs(np.linalg.eigvalsh(current_S))))
        local_best_signs = signs.copy()
        local_best_radius = current_radius
        
        start_time = time.time()
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() - start_time < max_time and no_improve_count < max_no_improve:
            improved = False
            
            # Single-flip pass
            for i in range(num_edges):
                if time.time() > deadline - 1.0:
                    return local_best_signs, local_best_radius
                
                old_sign = signs[i]
                new_sign = -old_sign
                signs[i] = new_sign
                
                if not is_lift_connected(signs):
                    signs[i] = old_sign
                    continue
                
                # Fast radius computation
                S_new = current_S.copy()
                u, v = edges[i]
                delta = new_sign - old_sign
                S_new[u, v] += delta
                S_new[v, u] += delta
                new_radius = float(np.max(np.abs(np.linalg.eigvalsh(S_new))))
                
                if new_radius < current_radius - 1e-12:
                    current_S = S_new
                    current_radius = new_radius
                    improved = True
                    no_improve_count = 0
                    if new_radius < local_best_radius:
                        local_best_radius = new_radius
                        local_best_signs = signs.copy()
                else:
                    signs[i] = old_sign
            
            # Double-flip pass to escape local minima
            if not improved:
                no_improve_count += 1
                if no_improve_count % 3 == 0:
                    for _ in range(num_edges):
                        if time.time() > deadline - 1.0:
                            return local_best_signs, local_best_radius
                        
                        i = rng.integers(0, num_edges)
                        j = rng.integers(0, num_edges)
                        if i == j:
                            continue
                        
                        old_si, old_sj = signs[i], signs[j]
                        signs[i] = -old_si
                        signs[j] = -old_sj
                        
                        if not is_lift_connected(signs):
                            signs[i] = old_si
                            signs[j] = old_sj
                            continue
                        
                        S_new = current_S.copy()
                        ui, vi = edges[i]
                        uj, vj = edges[j]
                        S_new[ui, vi] += 2 * (-old_si)
                        S_new[vi, ui] += 2 * (-old_si)
                        S_new[uj, vj] += 2 * (-old_sj)
                        S_new[vj, uj] += 2 * (-old_sj)
                        new_radius = float(np.max(np.abs(np.linalg.eigvalsh(S_new))))
                        
                        if new_radius < current_radius - 1e-12:
                            current_S = S_new
                            current_radius = new_radius
                            improved = True
                            no_improve_count = 0
                            if new_radius < local_best_radius:
                                local_best_radius = new_radius
                                local_best_signs = signs.copy()
                        else:
                            signs[i] = old_si
                            signs[j] = old_sj
            
            if not improved:
                no_improve_count += 1
        
        return local_best_signs, local_best_radius

    # Generate diverse initial patterns
    def make_pattern(top_frustration, bottom_frustration, match_pattern):
        """
        Create a sign pattern given cycle frustrations and matching pattern.
        """
        signs = np.ones(num_edges, dtype=np.int8)
        
        # Classify edges
        top_edges = []
        bottom_edges = []
        match_edges = []
        for idx, (u, v) in enumerate(edges):
            if u < m and v < m:
                top_edges.append(idx)
            elif u >= m and v >= m:
                bottom_edges.append(idx)
            else:
                match_edges.append(idx)
        
        if top_frustration == -1 and top_edges:
            signs[top_edges[0]] = -1
        if bottom_frustration == -1 and bottom_edges:
            signs[bottom_edges[0]] = -1
        
        if match_pattern == 'all_plus':
            pass
        elif match_pattern == 'all_minus':
            for idx in match_edges:
                signs[idx] = -1
        elif match_pattern == 'alternating':
            for i, idx in enumerate(match_edges):
                signs[idx] = 1 if i % 2 == 0 else -1
        elif match_pattern == 'alternating_shift':
            for i, idx in enumerate(match_edges):
                signs[idx] = -1 if i % 2 == 0 else 1
        
        return signs

    # Generate initial patterns
    patterns = []
    for tf in [1, -1]:
        for bf in [1, -1]:
            for mp in ['all_plus', 'all_minus', 'alternating', 'alternating_shift']:
                patterns.append(make_pattern(tf, bf, mp))
    
    # Add random patterns
    for _ in range(20):
        patterns.append(rng.choice([-1, 1], size=num_edges).astype(np.int8))
    
    # Run local search on each pattern
    for pattern in patterns:
        if time.time() > deadline - 15.0:
            break
        result_signs, result_radius = local_search(pattern, max_time=3.0)
        if result_signs is not None and result_radius < best_radius:
            best_radius = result_radius
            best_signs = result_signs.copy()
    
    # Strategy 3: Simulated Annealing to escape local minima
    if best_signs is not None and time.time() < deadline - 10.0:
        current_signs = best_signs.copy()
        current_S = build_matrix(current_signs)
        current_radius = float(np.max(np.abs(np.linalg.eigvalsh(current_S))))
        
        T = 0.1
        T_min = 0.001
        cooling = 0.98
        sa_start_time = time.time()
        
        while T > T_min and time.time() - sa_start_time < 12.0:
            if time.time() > deadline - 3.0:
                break
            
            i = rng.integers(0, num_edges)
            old_sign = current_signs[i]
            new_sign = -old_sign
            current_signs[i] = new_sign
            
            if not is_lift_connected(current_signs):
                current_signs[i] = old_sign
                T *= cooling
                continue
            
            S_new = current_S.copy()
            u, v = edges[i]
            delta = new_sign - old_sign
            S_new[u, v] += delta
            S_new[v, u] += delta
            new_radius = float(np.max(np.abs(np.linalg.eigvalsh(S_new))))
            
            delta_obj = new_radius - current_radius
            
            if delta_obj < 0 or rng.random() < np.exp(-delta_obj / T):
                current_S = S_new
                current_radius = new_radius
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_signs = current_signs.copy()
            else:
                current_signs[i] = old_sign
            
            T *= cooling
        
        # Polish with local search after SA
        if best_signs is not None:
            result_signs, result_radius = local_search(best_signs.copy(), max_time=3.0)
            if result_signs is not None and result_radius < best_radius:
                best_radius = result_radius
                best_signs = result_signs.copy()
    
    # Final high-precision polish: multiple rounds of single flips with tighter tolerance
    if best_signs is not None and time.time() < deadline - 2.0:
        for _ in range(5):
            if time.time() > deadline - 1.0:
                break
            improved = True
            while improved:
                improved = False
                for e in range(num_edges):
                    if time.time() > deadline - 0.5:
                        break
                    old_s = best_signs[e]
                    best_signs[e] = -old_s
                    
                    if not is_lift_connected(best_signs):
                        best_signs[e] = old_s
                        continue
                    
                    S = build_matrix(best_signs)
                    obj = float(np.max(np.abs(np.linalg.eigvalsh(S))))
                    
                    if obj < best_radius - 1e-14:
                        best_radius = obj
                        improved = True
                    else:
                        best_signs[e] = old_s
    
    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=np.int8)
    
    return best_signs.astype(np.int8)
