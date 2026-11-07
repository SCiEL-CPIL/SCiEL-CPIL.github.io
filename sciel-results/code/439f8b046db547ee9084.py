def sign_edges(graph, budget):
    import numpy as np
    import time
    import random
    from itertools import product

    start_time = time.time()
    DEADLINE = 88.0

    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m

    # Classify edges
    top_edges = []
    bot_edges = []
    match_edges = []
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_edges.append(i)
        elif u >= m and v >= m:
            bot_edges.append(i)
        else:
            match_edges.append(i)

    # Precompute edge arrays
    edge_u = np.array([e[0] for e in edges], dtype=int)
    edge_v = np.array([e[1] for e in edges], dtype=int)

    # Build adjacency list for validity check
    adj = [[] for _ in range(n_vertices)]
    for idx in range(n_edges):
        u, v = edge_u[idx], edge_v[idx]
        adj[u].append((idx, v))
        adj[v].append((idx, u))

    def spectral_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        A = np.zeros((n_vertices, n_vertices))
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        vals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(vals)))

    def is_valid_lift(signs):
        """Check if the 2-lift is connected (i.e., signed graph is unbalanced)."""
        n = n_vertices
        sigma = np.zeros(n, dtype=int)
        visited = np.zeros(n, dtype=bool)
        sigma[0] = 1
        visited[0] = True
        q = [0]
        head = 0

        while head < len(q):
            u = q[head]
            head += 1
            for edge_idx, v in adj[u]:
                s = signs[edge_idx]
                expected_sigma_v = s * sigma[u]
                if not visited[v]:
                    sigma[v] = expected_sigma_v
                    visited[v] = True
                    q.append(v)
                else:
                    if sigma[v] != expected_sigma_v:
                        return True  # Unbalanced, lift is connected

        return False  # Balanced, lift is disconnected

    def check_valid_and_radius(signs):
        """Returns (radius, is_valid)."""
        if not is_valid_lift(signs):
            return float('inf'), False
        return spectral_radius(signs), True

    best_signs = None
    best_radius = float('inf')

    # Strategy 1: Exhaustive search for small m
    if m <= 5:
        # For m=4: 4 top + 4 bottom + 4 match = 12 edges
        # For m=5: 5 top + 5 bottom + 5 match = 15 edges
        # 2^15 = 32768, which is feasible
        if n_edges <= 15:
            for mask in range(1 << n_edges):
                if (time.time() - start_time) > DEADLINE:
                    break
                signs = np.ones(n_edges, dtype=int)
                for i in range(n_edges):
                    if mask & (1 << i):
                        signs[i] = -1
                radius, valid = check_valid_and_radius(signs)
                if valid and radius < best_radius:
                    best_radius = radius
                    best_signs = signs.copy()

    # Strategy 2: Local search with multiple restarts
    # Use simulated annealing or simple hill climbing with random restarts

    def local_search(init_signs, max_iterations=10000):
        """Hill climbing with random restarts within a single run."""
        signs = init_signs.copy()
        current_radius, current_valid = check_valid_and_radius(signs)

        if not current_valid:
            # Try to make it valid by flipping random edges
            for _ in range(100):
                if (time.time() - start_time) > DEADLINE:
                    return None, float('inf')
                idx = random.randint(0, n_edges - 1)
                signs[idx] *= -1
                current_radius, current_valid = check_valid_and_radius(signs)
                if current_valid:
                    break
            if not current_valid:
                return None, float('inf')

        best_signs_local = signs.copy()
        best_radius_local = current_radius

        no_improve_count = 0
        for it in range(max_iterations):
            if (time.time() - start_time) > DEADLINE:
                break

            # Pick a random edge to flip
            idx = random.randint(0, n_edges - 1)
            old_sign = signs[idx]
            signs[idx] = -old_sign

            new_radius, new_valid = check_valid_and_radius(signs)

            if new_valid and new_radius < current_radius:
                current_radius = new_radius
                no_improve_count = 0
                if new_radius < best_radius_local:
                    best_radius_local = new_radius
                    best_signs_local = signs.copy()
            else:
                # Revert
                signs[idx] = old_sign
                no_improve_count += 1

            if no_improve_count > 50:
                # Stagnation: do a random perturbation
                for _ in range(3):
                    pert_idx = random.randint(0, n_edges - 1)
                    signs[pert_idx] *= -1
                current_radius, current_valid = check_valid_and_radius(signs)
                if not current_valid:
                    # Try to recover
                    for _ in range(20):
                        rec_idx = random.randint(0, n_edges - 1)
                        signs[rec_idx] *= -1
                        current_radius, current_valid = check_valid_and_radius(signs)
                        if current_valid:
                            break
                    if not current_valid:
                        return best_signs_local, best_radius_local
                no_improve_count = 0

        return best_signs_local, best_radius_local

    # Generate diverse initial candidates
    candidates = []

    # All +1 (likely invalid for even cycles, but check)
    s = np.ones(n_edges, dtype=int)
    candidates.append(s)

    # Alternating on top cycle
    s = np.ones(n_edges, dtype=int)
    for i, eidx in enumerate(top_edges):
        if i % 2 == 1:
            s[eidx] = -1
    candidates.append(s)

    # Alternating on bottom cycle
    s = np.ones(n_edges, dtype=int)
    for i, eidx in enumerate(bot_edges):
        if i % 2 == 1:
            s[eidx] = -1
    candidates.append(s)

    # Alternating on matching
    s = np.ones(n_edges, dtype=int)
    for i, eidx in enumerate(match_edges):
        if i % 2 == 1:
            s[eidx] = -1
    candidates.append(s)

    # One negative on each type
    for eidx in top_edges:
        s = np.ones(n_edges, dtype=int)
        s[eidx] = -1
        candidates.append(s)
    for eidx in bot_edges:
        s = np.ones(n_edges, dtype=int)
        s[eidx] = -1
        candidates.append(s)
    for eidx in match_edges:
        s = np.ones(n_edges, dtype=int)
        s[eidx] = -1
        candidates.append(s)

    # Random candidates
    rng = np.random.default_rng(42)
    for _ in range(20):
        s = rng.choice([-1, 1], size=n_edges).astype(int)
        candidates.append(s)

    # Run local search on each candidate
    for cand in candidates:
        if (time.time() - start_time) > DEADLINE - 1:
            break
        s, r = local_search(cand, max_iterations=5000)
        if s is not None and r < best_radius:
            best_radius = r
            best_signs = s

    # If we still have time, do more random restarts
    while (time.time() - start_time) < DEADLINE - 2:
        init = rng.choice([-1, 1], size=n_edges).astype(int)
        s, r = local_search(init, max_iterations=2000)
        if s is not None and r < best_radius:
            best_radius = r
            best_signs = s

    # Fallback if no valid solution found
    if best_signs is None:
        best_signs = np.ones(n_edges, dtype=int)
        if match_edges:
            best_signs[match_edges[0]] = -1
        # Ensure validity
        if not is_valid_lift(best_signs):
            # Flip more edges to make it unbalanced
            for i in range(n_edges):
                best_signs[i] *= -1
                if is_valid_lift(best_signs):
                    break

    return best_signs
