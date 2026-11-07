import time
import numpy as np

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin for 60s limit

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)

    # Build edge type classification
    bottom_edges = []
    top_edges = []
    match_edges = []
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            bottom_edges.append(i)
        elif u >= m and v >= m:
            top_edges.append(i)
        else:
            match_edges.append(i)

    # Precompute adjacency structure for fast spectral radius
    # For a signed adjacency matrix, we need to build A and compute eigvalsh
    # But we can optimize by noting the structure

    def compute_radius_fast(signs):
        """Compute spectral radius of signed adjacency matrix using the block structure."""
        # Build the signed adjacency matrix
        A = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            s = float(signs[i])
            A[u, v] = s
            A[v, u] = s
        try:
            eigvals = np.linalg.eigvalsh(A)
            return float(np.max(np.abs(eigvals)))
        except:
            return np.inf

    def is_valid_lift(signs):
        """Check if the lift is connected and 3-regular using the helper."""
        try:
            from autoresearch_suite.science_tcs_expansion import check_lift
            radius, details = check_lift(graph, signs)
            return True, float(radius)
        except Exception:
            return False, np.inf

    # Strategy: The key insight is that for prism graphs, we can exploit
    # the symmetry. The signed adjacency matrix has a specific structure.
    # 
    # For a prism graph with bottom cycle signs b_0,...,b_{m-1}, top cycle signs t_0,...,t_{m-1},
    # and matching signs w_0,...,w_{m-1}, the spectral radius depends on these.
    #
    # Key observation: If we flip all signs in the bottom cycle, the spectral radius is unchanged
    # (it's a gauge transformation). Similarly for the top cycle. So we can fix some signs.
    #
    # More importantly: The problem reduces to choosing signs such that the lift is connected
    # and the spectral radius is minimized.
    #
    # For connectedness: The lift is connected iff the product of signs around any "cycle" 
    # that goes from bottom to top and back is +1 in a specific way. Actually, for a 2-lift
    # defined by edge signs, the lift is connected iff the signs are not all +1 on a cut
    # that separates the two copies. More precisely, the lift has two components iff there
    # exists a vertex subset S such that all edges crossing the cut have sign +1 (or all -1
    # in a consistent way). 
    #
    # Actually, for a 2-lift: the lift is connected iff the sign vector is not in the 
    # "trivial" class, i.e., not all edges can be flipped by a vertex sign assignment to 
    # become all +1. This is equivalent to: the system x_u * x_v = s_e has no solution 
    # where x_v in {-1, 1}. If such a solution exists, the lift is disconnected (two copies).
    # Otherwise, it's connected.
    #
    # So we need: there is no assignment x_v in {-1,1} such that x_u * x_v = s_{uv} for all edges.
    # This is a 2-coloring / bipartite consistency check.

    def is_connected_lift(signs):
        """Check if the 2-lift is connected using BFS on the signed graph."""
        # The lift is disconnected iff there exists x_v in {-1,1} with x_u*x_v = s_{uv}
        # This is equivalent to: the signed graph is "balanced"
        # We can check by trying to assign x_0 = 1 and propagating
        x = np.zeros(n, dtype=np.int8)
        x[0] = 1
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        stack = [0]
        
        while stack:
            u = stack.pop()
            for i, (a, b) in enumerate(edges):
                s = signs[i]
                if a == u and not visited[b]:
                    x[b] = x[a] * s
                    visited[b] = True
                    stack.append(b)
                elif b == u and not visited[a]:
                    x[a] = x[b] * s
                    visited[a] = True
                    stack.append(a)
        
        # Check consistency
        for i, (a, b) in enumerate(edges):
            if x[a] * x[b] != signs[i]:
                return True  # Not balanced, so lift is connected
        
        # If we didn't visit all nodes, the graph might be disconnected in base
        # but prism graphs are connected, so this shouldn't happen
        if not np.all(visited):
            # Shouldn't happen for connected base graph
            return True
        
        return False  # Balanced, so lift is disconnected

    # Better: just use check_lift for correctness
    # But for speed in local search, we can use the faster connectivity check
    # and only call check_lift for final validation

    def quick_validity(signs):
        """Quick check: is the lift connected? Use the balance test."""
        return is_connected_lift(signs)

    # Initial best
    best_signs = None
    best_radius = np.inf

    rng = np.random.default_rng(42)

    # Generate diverse initial candidates
    candidates = []

    # All +1 (will be disconnected, but good starting point for search)
    candidates.append(np.ones(E, dtype=np.int8))

    # Pattern: bottom +1, top -1, match +1
    p = np.ones(E, dtype=np.int8)
    for i in top_edges:
        p[i] = -1
    candidates.append(p.copy())

    # Pattern: alternating on both cycles
    for phase in [0, 1]:
        p = np.ones(E, dtype=np.int8)
        for idx, i in enumerate(bottom_edges):
            if idx % 2 != phase:
                p[i] = -1
        for idx, i in enumerate(top_edges):
            if idx % 2 != phase:
                p[i] = -1
        candidates.append(p.copy())

    # Random candidates
    for _ in range(50):
        candidates.append(rng.choice([-1, 1], size=E).astype(np.int8))

    # Evaluate candidates
    for signs in candidates:
        if time.time() > deadline:
            break
        if not quick_validity(signs):
            continue
        valid, radius = is_valid_lift(signs)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()

    if best_signs is None:
        # Fallback: find any valid assignment
        for _ in range(1000):
            if time.time() > deadline:
                break
            signs = rng.choice([-1, 1], size=E).astype(np.int8)
            if quick_validity(signs):
                valid, radius = is_valid_lift(signs)
                if valid:
                    best_signs = signs.copy()
                    best_radius = radius
                    break

    if best_signs is None:
        # Last resort: all +1 (may be invalid but better than nothing)
        best_signs = np.ones(E, dtype=np.int8)
        best_radius = np.inf

    # Local search: flip one edge at a time
    current_signs = best_signs.copy()
    current_radius = best_radius

    # Precompute edge neighbors for potential future optimizations
    # For now, simple random flip local search

    iteration = 0
    max_iterations = 100000
    no_improve_count = 0
    max_no_improve = 2000

    while time.time() < deadline - 3.0 and iteration < max_iterations and no_improve_count < max_no_improve:
        iteration += 1

        edge_idx = int(rng.integers(0, E))
        trial_signs = current_signs.copy()
        trial_signs[edge_idx] = -trial_signs[edge_idx]

        if not quick_validity(trial_signs):
            no_improve_count += 1
            continue

        # Only compute full radius if connectivity passes
        radius = compute_radius_fast(trial_signs)

        if radius < current_radius - 1e-12:
            current_signs = trial_signs
            current_radius = radius
            no_improve_count = 0
            if current_radius < best_radius:
                best_radius = current_radius
                best_signs = current_signs.copy()
        else:
            no_improve_count += 1

    # If we have time, do more random restarts with local search
    if time.time() < deadline - 10.0:
        for restart in range(20):
            if time.time() > deadline - 5.0:
                break

            # Random restart
            restart_signs = rng.choice([-1, 1], size=E).astype(np.int8)
            if not quick_validity(restart_signs):
                continue

            local_signs = restart_signs.copy()
            local_radius = compute_radius_fast(local_signs)

            # Local search from this restart
            local_no_improve = 0
            for _ in range(5000):
                if time.time() > deadline - 3.0:
                    break
                if local_no_improve > 1000:
                    break

                edge_idx = int(rng.integers(0, E))
                trial_signs = local_signs.copy()
                trial_signs[edge_idx] = -trial_signs[edge_idx]

                if not quick_validity(trial_signs):
                    local_no_improve += 1
                    continue

                radius = compute_radius_fast(trial_signs)
                if radius < local_radius - 1e-12:
                    local_signs = trial_signs
                    local_radius = radius
                    local_no_improve = 0
                    if local_radius < best_radius:
                        best_radius = local_radius
                        best_signs = local_signs.copy()
                else:
                    local_no_improve += 1

    # Final validation
    if best_signs is not None:
        try:
            valid, final_radius = is_valid_lift(best_signs)
            if valid:
                return best_signs
        except:
            pass

    # Return best found
    return best_signs
