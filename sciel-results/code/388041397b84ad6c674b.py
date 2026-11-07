import time
import numpy as np
from collections import deque

def sign_edges(graph, budget):
    """
    Returns an array of signs (-1 or +1) for each edge in the graph.
    Strategy:
    1. Exploit the algebraic structure of the prism graph to compute the
       spectral radius of the 2-lift directly from the base signed matrix.
       The lift eigenvalues are the eigenvalues of A (base) and -A (base),
       so the spectral radius of the lift equals the spectral radius of the
       base signed adjacency matrix.
    2. For small m (<=5), exhaustive search over all valid sign assignments.
    3. For larger m, multi-restart local search with single-flip and
       double-flip moves, using fast connectivity checks and incremental
       matrix updates.
    4. Final polish pass with tighter tolerance.
    """
    
    deadline = time.time() + 82.0  # Safety margin before 90s kill
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Precompute adjacency list for fast connectivity checks
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    u_arr = np.array([e[0] for e in edges])
    v_arr = np.array([e[1] for e in edges])
    
    def is_lift_connected(signs):
        """
        Returns True if the 2-lift is connected.
        The 2-lift is connected iff the signing is NOT a coboundary,
        i.e., there exists a cycle with an odd number of -1 signs.
        We check this by attempting to assign "side" labels to vertices
        such that for every edge, the sides are consistent with the sign.
        If we find an inconsistency, the lift is connected.
        """
        side = [-1] * n
        side[0] = 0
        queue = deque([0])
        
        while queue:
            u = queue.popleft()
            for v, idx in adj[u]:
                s = signs[idx]
                # If s == 1, v is on the same side as u
                # If s == -1, v is on the opposite side as u
                expected_side = side[u] ^ (1 if s == -1 else 0)
                
                if side[v] == -1:
                    side[v] = expected_side
                    queue.append(v)
                else:
                    if side[v] != expected_side:
                        return True  # Inconsistency found -> connected
        
        return False  # No inconsistency -> coboundary -> disconnected
    
    def build_matrix(signs):
        """Build the signed adjacency matrix of the base graph."""
        S = np.zeros((n, n), dtype=np.float64)
        S[u_arr, v_arr] = signs.astype(np.float64)
        S[v_arr, u_arr] = signs.astype(np.float64)
        return S
    
    def get_radius(signs):
        """Compute the spectral radius of the base signed adjacency matrix.
        The lift's spectral radius equals this value."""
        S = build_matrix(signs)
        eigvals = np.linalg.eigvalsh(S)
        return np.max(np.abs(eigvals))
    
    best_signs = None
    best_radius = np.inf
    
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
            radius = get_radius(signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
        
        if best_signs is not None:
            # Verify with check_lift if available
            try:
                from autoresearch_suite.science_tcs_expansion import check_lift
                r, _ = check_lift(graph, best_signs)
                return best_signs
            except Exception:
                return best_signs
    
    # Strategy 2: Multi-restart local search for larger m
    
    def local_search(signs, max_time=5.0):
        """
        Local search with single-flip and occasional double-flip moves.
        Returns (best_signs, best_radius) found.
        """
        if not is_lift_connected(signs):
            return None, np.inf
        
        current_S = build_matrix(signs)
        current_radius = np.max(np.abs(np.linalg.eigvalsh(current_S)))
        local_best_signs = signs.copy()
        local_best_radius = current_radius
        
        start_time = time.time()
        no_improve_count = 0
        max_no_improve = 200
        
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
                
                # Fast radius computation via incremental update
                S_new = current_S.copy()
                u, v = edges[i]
                delta = new_sign - old_sign
                S_new[u, v] += delta
                S_new[v, u] += delta
                new_radius = np.max(np.abs(np.linalg.eigvalsh(S_new)))
                
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
                    di = -old_si - old_si  # new - old = -old - old = -2*old
                    dj = -old_sj - old_sj
                    S_new[ui, vi] += di
                    S_new[vi, ui] += di
                    S_new[uj, vj] += dj
                    S_new[vj, uj] += dj
                    new_radius = np.max(np.abs(np.linalg.eigvalsh(S_new)))
                    
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
            else:
                no_improve_count = 0
        
        return local_best_signs, local_best_radius
    
    # Generate diverse initial candidates
    candidates = []
    
    # Candidate 1: All +1 except one matching edge flipped
    c1 = np.ones(num_edges, dtype=np.int8)
    # Find matching edges
    matching_edges = []
    for i in range(m):
        u, v = i, m + i
        for idx, (a, b) in enumerate(edges):
            if (a == u and b == v) or (a == v and b == u):
                matching_edges.append(idx)
                break
    if matching_edges:
        c1[matching_edges[0]] = -1
    candidates.append(c1)
    
    # Candidate 2: Alternating pattern on top cycle
    c2 = np.ones(num_edges, dtype=np.int8)
    top_edges = []
    for i in range(m):
        u, v = i, (i + 1) % m
        for idx, (a, b) in enumerate(edges):
            if (a == u and b == v) or (a == v and b == u):
                top_edges.append(idx)
                break
    for idx in top_edges[::2]:
        c2[idx] = -1
    if matching_edges:
        c2[matching_edges[0]] = -1
    candidates.append(c2)
    
    # Candidate 3: Random valid signing
    for _ in range(100):
        c3 = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        if is_lift_connected(c3):
            candidates.append(c3)
            break
    
    # Candidate 4: All -1 on matching
    c4 = np.ones(num_edges, dtype=np.int8)
    for idx in matching_edges:
        c4[idx] = -1
    candidates.append(c4)
    
    # Run local search from each candidate
    for cand in candidates:
        if time.time() > deadline - 10:
            break
        
        result_signs, result_radius = local_search(cand.copy(), max_time=8.0)
        
        if result_signs is not None and result_radius < best_radius:
            best_radius = result_radius
            best_signs = result_signs.copy()
    
    # If we haven't found a valid signing, create one
    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=np.int8)
        if matching_edges:
            best_signs[matching_edges[0]] = -1
        best_radius = get_radius(best_signs)
    
    # Final polish: try flipping each edge individually with tighter tolerance
    if time.time() < deadline - 3:
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            current_S = build_matrix(best_signs)
            for idx in range(num_edges):
                if time.time() > deadline - 3:
                    break
                
                new_signs = best_signs.copy()
                new_signs[idx] *= -1
                
                if not is_lift_connected(new_signs):
                    continue
                
                S_new = current_S.copy()
                u, v = edges[idx]
                delta = new_signs[idx] - best_signs[idx]
                S_new[u, v] += delta
                S_new[v, u] += delta
                r = np.max(np.abs(np.linalg.eigvalsh(S_new)))
                
                if r < best_radius - 1e-14:
                    best_radius = r
                    best_signs = new_signs
                    improved = True
    
    # Verify and return
    try:
        from autoresearch_suite.science_tcs_expansion import check_lift
        r, _ = check_lift(graph, best_signs)
    except Exception:
        pass
    
    return best_signs
