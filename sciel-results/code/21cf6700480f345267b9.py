import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    start_time = time.time()
    DEADLINE = 88  # Safety margin
    
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m
    
    # Precompute edge endpoints
    edge_u = np.array([e[0] for e in edges], dtype=int)
    edge_v = np.array([e[1] for e in edges], dtype=int)
    
    # Precompute adjacency list for BFS
    adj = [[] for _ in range(n_vertices)]
    for idx in range(n_edges):
        u, v = edge_u[idx], edge_v[idx]
        adj[u].append((idx, v))
        adj[v].append((idx, u))
    
    def is_unbalanced(signs):
        """
        Returns True if the signed graph is unbalanced (lift is connected).
        A graph is balanced if there exists sigma: V -> {+1, -1} such that 
        s(u,v) = sigma(u)*sigma(v) for all edges.
        """
        sigma = np.zeros(n_vertices, dtype=int)
        visited = np.zeros(n_vertices, dtype=bool)
        
        # The graph is connected (prism graphs are connected), so we only need one BFS
        sigma[0] = 1
        visited[0] = True
        q = deque([0])
        
        while q:
            u = q.popleft()
            for edge_idx, v in adj[u]:
                s = signs[edge_idx]
                expected_sigma_v = s * sigma[u]
                if not visited[v]:
                    sigma[v] = expected_sigma_v
                    visited[v] = True
                    q.append(v)
                else:
                    if sigma[v] != expected_sigma_v:
                        return True  # Contradiction found, unbalanced
        
        return False  # No contradiction, balanced
    
    def build_matrix(signs):
        A = np.zeros((n_vertices, n_vertices))
        for idx in range(n_edges):
            s = signs[idx]
            u, v = edge_u[idx], edge_v[idx]
            A[u, v] = s
            A[v, u] = s
        return A

    def spectral_radius(signs):
        A = build_matrix(signs)
        vals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(vals)))

    def local_search(initial_signs, max_time_remaining):
        signs = initial_signs.copy()
        
        # Ensure validity
        if not is_unbalanced(signs):
            # Flip random edges until valid
            for _ in range(100):
                idx = np.random.randint(n_edges)
                signs[idx] *= -1
                if is_unbalanced(signs):
                    break
            if not is_unbalanced(signs):
                return None, float('inf')
        
        A = build_matrix(signs)
        current_radius = spectral_radius(signs)
        best_signs = signs.copy()
        best_radius = current_radius
        
        no_improve_count = 0
        while time.time() - start_time < max_time_remaining:
            improved = False
            
            # 1-flip moves
            for i in range(n_edges):
                if time.time() - start_time > max_time_remaining:
                    break
                
                old_sign = signs[i]
                signs[i] = -old_sign
                
                # Check validity
                if is_unbalanced(signs):
                    # Update matrix
                    u, v = edge_u[i], edge_v[i]
                    A[u, v] = -A[u, v]
                    A[v, u] = -A[v, u]
                    
                    new_radius = spectral_radius(signs)
                    if new_radius < current_radius - 1e-10:
                        current_radius = new_radius
                        improved = True
                        no_improve_count = 0
                        best_signs = signs.copy()
                        best_radius = current_radius
                    else:
                        # Revert
                        signs[i] = old_sign
                        A[u, v] = -A[u, v]
                        A[v, u] = -A[v, u]
                else:
                    # Revert
                    signs[i] = old_sign
            
            if improved:
                continue
            
            no_improve_count += 1
            
            # 2-flip moves if stuck
            if no_improve_count > 2:
                for i in range(n_edges):
                    if time.time() - start_time > max_time_remaining:
                        break
                    for j in range(i + 1, n_edges):
                        if time.time() - start_time > max_time_remaining:
                            break
                        
                        si, sj = signs[i], signs[j]
                        signs[i] = -si
                        signs[j] = -sj
                        
                        if is_unbalanced(signs):
                            ui, vi = edge_u[i], edge_v[i]
                            uj, vj = edge_u[j], edge_v[j]
                            A[ui, vi] = -A[ui, vi]
                            A[vi, ui] = -A[vi, ui]
                            A[uj, vj] = -A[uj, vj]
                            A[vj, uj] = -A[vj, uj]
                            
                            new_radius = spectral_radius(signs)
                            if new_radius < current_radius - 1e-10:
                                current_radius = new_radius
                                improved = True
                                no_improve_count = 0
                                best_signs = signs.copy()
                                best_radius = current_radius
                            else:
                                signs[i] = si
                                signs[j] = sj
                                A[ui, vi] = -A[ui, vi]
                                A[vi, ui] = -A[vi, ui]
                                A[uj, vj] = -A[uj, vj]
                                A[vj, uj] = -A[vj, uj]
                        else:
                            signs[i] = si
                            signs[j] = sj
            
            if improved:
                continue
            
            # If stuck for too long, break to restart
            if no_improve_count > 10:
                break
        
        return best_signs, best_radius

    # Define structured initializations
    top_edges = []
    bot_edges = []
    match_edges = []
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_edges.append(idx)
        elif u >= m and v >= m:
            bot_edges.append(idx)
        else:
            match_edges.append(idx)
    
    initializations = []
    
    # 1. All +1 (likely balanced, will be fixed)
    initializations.append(np.ones(n_edges, dtype=int))
    
    # 2. Alternating on top
    s = np.ones(n_edges, dtype=int)
    for i, eidx in enumerate(top_edges):
        if i % 2 == 1:
            s[eidx] = -1
    initializations.append(s)
    
    # 3. Alternating on bottom
    s = np.ones(n_edges, dtype=int)
    for i, eidx in enumerate(bot_edges):
        if i % 2 == 1:
            s[eidx] = -1
    initializations.append(s)
    
    # 4. Alternating on matching
    s = np.ones(n_edges, dtype=int)
    for i, eidx in enumerate(match_edges):
        if i % 2 == 1:
            s[eidx] = -1
    initializations.append(s)
    
    # 5. Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        initializations.append(rng.choice([-1, 1], size=n_edges).astype(int))
        
    # 6. Specific patterns known to be unbalanced
    # Single -1 in top cycle
    s = np.ones(n_edges, dtype=int)
    if top_edges:
        s[top_edges[0]] = -1
    initializations.append(s)
    
    # Single -1 in bottom cycle
    s = np.ones(n_edges, dtype=int)
    if bot_edges:
        s[bot_edges[0]] = -1
    initializations.append(s)
    
    # Single -1 in matching
    s = np.ones(n_edges, dtype=int)
    if match_edges:
        s[match_edges[0]] = -1
    initializations.append(s)

    best_signs = None
    best_radius = float('inf')
    
    # Run local search on each initialization
    for init in initializations:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        s, r = local_search(init, DEADLINE - 2)
        if s is not None and r < best_radius:
            best_signs = s
            best_radius = r
    
    # If time remains, do more random restarts
    while time.time() - start_time < DEADLINE - 2:
        init = rng.choice([-1, 1], size=n_edges).astype(int)
        s, r = local_search(init, DEADLINE - 2)
        if s is not None and r < best_radius:
            best_signs = s
            best_radius = r
    
    if best_signs is None:
        # Fallback
        best_signs = np.ones(n_edges, dtype=int)
        if match_edges:
            best_signs[match_edges[0]] = -1
        # Ensure validity
        if not is_unbalanced(best_signs):
            best_signs[0] = -1
            if not is_unbalanced(best_signs):
                best_signs[1] = -1

    return best_signs
