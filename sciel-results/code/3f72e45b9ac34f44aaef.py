"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    DEADLINE = 85  # Safety margin before 90s kill

    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m

    from autoresearch_suite.science_tcs_expansion import check_lift

    # Build edge map: idx -> (type, pos)
    edge_map = {}
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            if v == (u + 1) % m:
                pos = u
            else:
                pos = v
            edge_map[idx] = ('top', pos)
        elif u >= m and v >= m:
            u_local = u - m
            v_local = v - m
            if v_local == (u_local + 1) % m:
                pos = u_local
            else:
                pos = v_local
            edge_map[idx] = ('bottom', pos)
        else:
            if u < m:
                pos = u
            else:
                pos = v - m
            edge_map[idx] = ('match', pos)

    # Build adjacency structure for fast spectral radius computation
    # Signed adjacency matrix A where A[u,v] = sign on edge (u,v)
    # We need to compute max |eigenvalue| of the signed adjacency matrix

    def build_signed_adj(signs):
        """Build the signed adjacency matrix from signs."""
        A = np.zeros((n_vertices, n_vertices), dtype=np.float64)
        for idx, (u, v) in enumerate(edges):
            A[u, v] = signs[idx]
            A[v, u] = signs[idx]
        return A

    def spectral_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        A = build_signed_adj(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))

    def is_connected_lift(signs):
        """Check if the 2-lift defined by signs is connected."""
        # In the 2-lift, vertex (v, s) for s in {0,1}
        # Edge (u,v) with sign +1 connects (u,0)-(v,0) and (u,1)-(v,1)
        # Edge (u,v) with sign -1 connects (u,0)-(v,1) and (u,1)-(v,0)
        # BFS from (0,0)
        visited = [[False, False] for _ in range(n_vertices)]
        visited[0][0] = True
        queue = [(0, 0)]
        count = 1
        while queue:
            v, s = queue.pop()
            for idx, (u, w) in enumerate(edges):
                if u == v:
                    other = w
                elif w == v:
                    other = u
                else:
                    continue
                sign = signs[idx]
                if sign == 1:
                    new_s = s
                else:
                    new_s = 1 - s
                if not visited[other][new_s]:
                    visited[other][new_s] = True
                    count += 1
                    queue.append((other, new_s))
        return count == 2 * n_vertices

    def evaluate(signs):
        """Evaluate a sign assignment. Returns spectral radius if valid, inf otherwise."""
        if not is_connected_lift(signs):
            return float('inf')
        return spectral_radius(signs)

    best_radius = float('inf')
    best_signs = None

    def try_candidate(signs):
        nonlocal best_radius, best_signs
        if time.time() > DEADLINE:
            return False
        r = evaluate(signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()
        return True

    # Phase 1: Analytic initialization - maximize frustration
    # For prism graphs, we want to make as many cycles as possible have product -1
    # The key cycles are: top cycle, bottom cycle, and the m "ladder" cycles (i, i+1, i+1+m, i+m)
    
    # Strategy: Use a checkerboard-like pattern that frustrates all ladder cycles
    # If top[i] * bottom[i] * match[i] * match[i+1] = -1 for all i, all ladder cycles are frustrated
    
    # Let's try: all match signs = +1, top signs alternate, bottom signs alternate
    # Then ladder cycle product = top[i] * bottom[i] * 1 * 1 = top[i]*bottom[i]
    # If top and bottom have same pattern, product = +1 (not frustrated)
    # If top and bottom have opposite patterns, product = -1 (frustrated!)
    
    # For even m: top = [+1,-1,+1,-1,...], bottom = [-1,+1,-1,+1,...]
    # For odd m: can't perfectly alternate, need one mismatch
    
    # Try multiple structured patterns
    patterns = []
    
    # Pattern A: Checkerboard on cycles
    if m % 2 == 0:
        top_a = np.array([1 if i % 2 == 0 else -1 for i in range(m)])
        bottom_a = np.array([-1 if i % 2 == 0 else 1 for i in range(m)])
        match_a = np.ones(m, dtype=int)
        patterns.append((top_a, bottom_a, match_a))
        
        top_b = np.array([-1 if i % 2 == 0 else 1 for i in range(m)])
        bottom_b = np.array([1 if i % 2 == 0 else -1 for i in range(m)])
        match_b = np.ones(m, dtype=int)
        patterns.append((top_b, bottom_b, match_b))
    else:
        # For odd m, use m-1 alternating and one +1
        for break_pos in range(m):
            top_c = np.array([1 if i % 2 == 0 else -1 for i in range(m)])
            top_c[break_pos] = 1
            bottom_c = np.array([-1 if i % 2 == 0 else 1 for i in range(m)])
            bottom_c[break_pos] = -1
            match_c = np.ones(m, dtype=int)
            patterns.append((top_c, bottom_c, match_c))
    
    # Pattern B: All matches -1, cycles +1
    top_d = np.ones(m, dtype=int)
    bottom_d = np.ones(m, dtype=int)
    match_d = np.ones(m, dtype=int)
    match_d[:] = -1
    patterns.append((top_d, bottom_d, match_d))
    
    # Pattern C: Mix - some matches -1
    for num_neg in range(1, min(m, 5)):
        for neg_pos in range(m):
            top_e = np.ones(m, dtype=int)
            bottom_e = np.ones(m, dtype=int)
            match_e = np.ones(m, dtype=int)
            match_e[neg_pos] = -1
            patterns.append((top_e, bottom_e, match_e))
    
    # Pattern D: Top cycle has k -1's, bottom has k -1's, matches all +1
    for k in [1, 2, 3]:
        if k <= m:
            for pos in range(m):
                top_f = np.ones(m, dtype=int)
                bottom_f = np.ones(m, dtype=int)
                match_f = np.ones(m, dtype=int)
                top_f[pos] = -1
                bottom_f[pos] = -1
                patterns.append((top_f, bottom_f, match_f))
    
    # Pattern E: Top cycle -1 at pos, bottom cycle -1 at (pos+1)%m
    for pos in range(m):
        top_g = np.ones(m, dtype=int)
        bottom_g = np.ones(m, dtype=int)
        match_g = np.ones(m, dtype=int)
        top_g[pos] = -1
        bottom_g[(pos + 1) % m] = -1
        patterns.append((top_g, bottom_g, match_g))
    
    # Evaluate all patterns
    for top_p, bottom_p, match_p in patterns:
        signs = np.ones(n_edges, dtype=int)
        for idx in range(n_edges):
            etype, pos = edge_map[idx]
            if etype == 'top':
                signs[idx] = top_p[pos]
            elif etype == 'bottom':
                signs[idx] = bottom_p[pos]
            else:
                signs[idx] = match_p[pos]
        try_candidate(signs)
    
    # Phase 2: Local search - flip individual edges and check improvement
    def local_search(signs, max_iterations=5000):
        nonlocal best_radius, best_signs
        current = signs.copy()
        current_radius = evaluate(current)
        if current_radius == float('inf'):
            return
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = current.copy()
        
        no_improve_count = 0
        for iteration in range(max_iterations):
            if time.time() > DEADLINE:
                return
            improved = False
            # Try flipping each edge
            for idx in range(n_edges):
                if time.time() > DEADLINE:
                    return
                current[idx] = -current[idx]
                new_radius = evaluate(current)
                if new_radius < current_radius:
                    current_radius = new_radius
                    improved = True
                    no_improve_count = 0
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_signs = current.copy()
                else:
                    current[idx] = -current[idx]  # revert
            if not improved:
                # Try 2-edge flips
                found_pair = False
                for i in range(n_edges):
                    if time.time() > DEADLINE:
                        return
                    for j in range(i + 1, n_edges):
                        if time.time() > DEADLINE:
                            return
                        current[i] = -current[i]
                        current[j] = -current[j]
                        new_radius = evaluate(current)
                        if new_radius < current_radius:
                            current_radius = new_radius
                            found_pair = True
                            no_improve_count = 0
                            if current_radius < best_radius:
                                best_radius = current_radius
                                best_signs = current.copy()
                            break
                        current[i] = -current[i]
                        current[j] = -current[j]
                    if found_pair:
                        break
                if not found_pair:
                    no_improve_count += 1
                    if no_improve_count >= 3:
                        break
        return current
    
    # Run local search from best found so far
    if best_signs is not None:
        improved = local_search(best_signs, max_iterations=2000)
    
    # Phase 3: More diverse starting points + local search
    # Random restarts with local search
    rng = np.random.default_rng(3)
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > DEADLINE:
            break
        # Generate random signs
        signs = rng.choice([-1, 1], size=n_edges)
        r = evaluate(signs)
        if r < float('inf'):
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
            # Local search from this point
            improved = local_search(signs, max_iterations=1000)
            if improved is not None:
                r2 = evaluate(improved)
                if r2 < best_radius:
                    best_radius = r2
                    best_signs = improved.copy()
    
    # Phase 4: Simulated annealing-like search
    if best_signs is not None and time.time() < DEADLINE - 5:
        current = best_signs.copy()
        current_radius = evaluate(current)
        T = 0.1
        for iteration in range(5000):
            if time.time() > DEADLINE:
                break
            idx = rng.integers(0, n_edges)
            current[idx] = -current[idx]
            new_radius = evaluate(current)
            delta = new_radius - current_radius
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_radius = new_radius
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_signs = current.copy()
            else:
                current[idx] = -current[idx]
            T *= 0.999
    
    if best_signs is None:
        # Fallback: all +1 with one -1
        best_signs = np.ones(n_edges, dtype=int)
        best_signs[0] = -1
    
    return best_signs
# EVOLVE-BLOCK-END
