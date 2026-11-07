def sign_edges(graph, budget):
    import numpy as np
    import time
    from collections import deque

    start_time = time.time()
    DEADLINE = 85  # Safety margin before 90s kill

    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m
    n_lift = 2 * n_vertices

    # Classify edges: cycle1 (0..m-1), cycle2 (m..2m-1), vertical
    # Prism graph: two m-cycles joined by matching (i, i+m)
    edge_types = []  # 0=cycle1, 1=cycle2, 2=vertical
    for (u, v) in edges:
        if u < m and v < m:
            edge_types.append(0)
        elif u >= m and v >= m:
            edge_types.append(1)
        else:
            edge_types.append(2)

    def build_signed_adj(signs):
        A = np.zeros((n_vertices, n_vertices))
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        return A

    def spectral_radius_from_A(A):
        return np.max(np.abs(np.linalg.eigvalsh(A)))

    def is_valid_lift_fast(signs):
        """Check if the 2-lift is connected using parity-based cycle analysis.
        
        For a prism graph, the lift is connected iff the product of signs
        around any cycle that connects the two sheets is -1 (odd number of -1s
        on a path connecting the cycles, or more generally, the holonomy
        around a cycle linking the two sheets is non-trivial).
        
        Actually, for a 2-lift of a graph, the lift is connected iff the
        signs define a non-trivial homomorphism to Z_2, i.e., not all cycles
        have product +1. Equivalently, there exists at least one cycle with
        product -1.
        
        For prism graphs specifically: the graph has two m-cycles and a matching.
        A fundamental cycle goes: around cycle1, across a vertical edge, around
        cycle2 in reverse, across another vertical edge back. The product of signs
        on this cycle determines connectivity.
        
        More simply: do BFS on the lift.
        """
        adj = [[] for _ in range(n_lift)]
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                adj[2*u].append(2*v)
                adj[2*v].append(2*u)
                adj[2*u+1].append(2*v+1)
                adj[2*v+1].append(2*u+1)
            else:
                adj[2*u].append(2*v+1)
                adj[2*v+1].append(2*u)
                adj[2*u+1].append(2*v)
                adj[2*v].append(2*u+1)

        visited = [False] * n_lift
        queue = deque([0])
        visited[0] = True
        count = 1
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    count += 1
                    queue.append(nb)
        return count == n_lift

    def make_valid(signs, rng):
        """Ensure signs produce a connected lift."""
        if is_valid_lift_fast(signs):
            return signs
        # Flip random edges until valid
        for _ in range(50):
            idx = rng.integers(0, n_edges)
            signs[idx] *= -1
            if is_valid_lift_fast(signs):
                return signs
        # Try flipping each single edge
        for i in range(n_edges):
            signs[i] *= -1
            if is_valid_lift_fast(signs):
                return signs
            signs[i] *= -1
        # Try flipping pairs
        for i in range(n_edges):
            for j in range(i+1, n_edges):
                signs[i] *= -1
                signs[j] *= -1
                if is_valid_lift_fast(signs):
                    return signs
                signs[i] *= -1
                signs[j] *= -1
        # Fallback: all +1 except one vertical edge
        signs = np.ones(n_edges, dtype=int)
        for i, t in enumerate(edge_types):
            if t == 2:
                signs[i] = -1
                break
        return signs

    best_signs = None
    best_radius = float('inf')

    rng = np.random.default_rng(3)

    # Strategy: use spectral radius as objective, local search with 1-flips
    # Key insight: for signed graphs, the spectral radius relates to the
    # structure. We want to minimize it.

    # Initial diverse starting points
    initial_signs_list = []

    # 1. All +1 (may need fixing)
    s = np.ones(n_edges, dtype=int)
    initial_signs_list.append(s)

    # 2. Random
    initial_signs_list.append(rng.choice([-1, 1], size=n_edges))

    # 3. Alternate on cycle 1
    s = np.ones(n_edges, dtype=int)
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            s[i] = 1 if (min(u, v) % 2 == 0) else -1
    initial_signs_list.append(s)

    # 4. All -1 on verticals
    s = np.ones(n_edges, dtype=int)
    for i, t in enumerate(edge_types):
        if t == 2:
            s[i] = -1
    initial_signs_list.append(s)

    # 5. All -1 on cycle 1
    s = np.ones(n_edges, dtype=int)
    for i, t in enumerate(edge_types):
        if t == 0:
            s[i] = -1
    initial_signs_list.append(s)

    # 6. All -1 on cycle 2
    s = np.ones(n_edges, dtype=int)
    for i, t in enumerate(edge_types):
        if t == 1:
            s[i] = -1
    initial_signs_list.append(s)

    # 7. Checkerboard-like pattern
    s = np.ones(n_edges, dtype=int)
    for i, (u, v) in enumerate(edges):
        s[i] = 1 if ((u + v) % 2 == 0) else -1
    initial_signs_list.append(s)

    # 8. Random with bias toward -1
    s = np.where(rng.random(n_edges) < 0.6, -1, 1).astype(int)
    initial_signs_list.append(s)

    # 9. Another random
    initial_signs_list.append(rng.choice([-1, 1], size=n_edges))

    # 10. Pattern: -1 on every other vertical
    s = np.ones(n_edges, dtype=int)
    vert_count = 0
    for i, t in enumerate(edge_types):
        if t == 2:
            if vert_count % 2 == 0:
                s[i] = -1
            vert_count += 1
    initial_signs_list.append(s)

    for init_signs in initial_signs_list:
        if time.time() - start_time > DEADLINE - 3:
            break

        signs = init_signs.copy()
        signs = make_valid(signs, rng)

        if not is_valid_lift_fast(signs):
            continue

        A = build_signed_adj(signs)
        current_radius = spectral_radius_from_A(A)

        # Local search: 1-flip
        improved = True
        while improved and (time.time() - start_time < DEADLINE - 1):
            improved = False
            # Try all 1-flips
            for i in range(n_edges):
                if time.time() - start_time > DEADLINE - 1:
                    break
                old = signs[i]
                u, v = edges[i]
                signs[i] = -old
                A[u, v] = -A[u, v]
                A[v, u] = -A[v, u]

                if not is_valid_lift_fast(signs):
                    signs[i] = old
                    A[u, v] = -A[u, v]
                    A[v, u] = -A[v, u]
                    continue

                new_radius = spectral_radius_from_A(A)
                if new_radius < current_radius - 1e-10:
                    current_radius = new_radius
                    improved = True
                else:
                    signs[i] = old
                    A[u, v] = -A[u, v]
                    A[v, u] = -A[v, u]

            # If no 1-flip helped, try random 2-flips
            if not improved:
                for _ in range(30):
                    if time.time() - start_time > DEADLINE - 1:
                        break
                    i, j = rng.choice(n_edges, size=2, replace=False)
                    ui, vi = edges[i]
                    uj, vj = edges[j]
                    oi = signs[i]
                    oj = signs[j]
                    signs[i] = -oi
                    signs[j] = -oj
                    A[ui, vi] = -A[ui, vi]
                    A[vi, ui] = -A[vi, ui]
                    A[uj, vj] = -A[uj, vj]
                    A[vj, uj] = -A[vj, uj]

                    if not is_valid_lift_fast(signs):
                        signs[i] = oi
                        signs[j] = oj
                        A[ui, vi] = -A[ui, vi]
                        A[vi, ui] = -A[vi, ui]
                        A[uj, vj] = -A[uj, vj]
                        A[vj, uj] = -A[vj, uj]
                        continue

                    new_radius = spectral_radius_from_A(A)
                    if new_radius < current_radius - 1e-10:
                        current_radius = new_radius
                        improved = True
                    else:
                        signs[i] = oi
                        signs[j] = oj
                        A[ui, vi] = -A[ui, vi]
                        A[vi, ui] = -A[vi, ui]
                        A[uj, vj] = -A[uj, vj]
                        A[vj, uj] = -A[vj, uj]

        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = signs.copy()

    # Continue with perturbation-based search from best
    if best_signs is not None and time.time() - start_time < DEADLINE - 2:
        current_signs = best_signs.copy()
        A = build_signed_adj(current_signs)
        current_radius = best_radius

        # More aggressive local search with 2-flips and 3-flips
        for _ in range(200):
            if time.time() - start_time > DEADLINE - 1:
                break

            # Try random 2-flips
            i, j = rng.choice(n_edges, size=2, replace=False)
            ui, vi = edges[i]
            uj, vj = edges[j]
            oi = current_signs[i]
            oj = current_signs[j]
            current_signs[i] = -oi
            current_signs[j] = -oj
            A[ui, vi] = -A[ui, vi]
            A[vi, ui] = -A[vi, ui]
            A[uj, vj] = -A[uj, vj]
            A[vj, uj] = -A[vj, uj]

            if not is_valid_lift_fast(current_signs):
                current_signs[i] = oi
                current_signs[j] = oj
                A[ui, vi] = -A[ui, vi]
                A[vi, ui] = -A[vi, ui]
                A[uj, vj] = -A[uj, vj]
                A[vj, uj] = -A[vj, uj]
                continue

            new_radius = spectral_radius_from_A(A)
            if new_radius < current_radius - 1e-10:
                current_radius = new_radius
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_signs = current_signs.copy()
            else:
                current_signs[i] = oi
                current_signs[j] = oj
                A[ui, vi] = -A[ui, vi]
                A[vi, ui] = -A[vi, ui]
                A[uj, vj] = -A[uj, vj]
                A[vj, uj] = -A[vj, uj]

            # Occasionally try 3-flips
            if rng.random() < 0.2:
                idxs = rng.choice(n_edges, size=3, replace=False)
                old_signs = current_signs[idxs].copy()
                for idx in idxs:
                    current_signs[idx] *= -1
                    u, v = edges[idx]
                    A[u, v] = -A[u, v]
                    A[v, u] = -A[v, u]

                if not is_valid_lift_fast(current_signs):
                    for idx in idxs:
                        current_signs[idx] = old_signs[list(idxs).index(idx)]
                        u, v = edges[idx]
                        A[u, v] = -A[u, v]
                        A[v, u] = -A[v, u]
                    continue

                new_radius = spectral_radius_from_A(A)
                if new_radius < current_radius - 1e-10:
                    current_radius = new_radius
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_signs = current_signs.copy()
                else:
                    for idx in idxs:
                        current_signs[idx] = old_signs[list(idxs).index(idx)]
                        u, v = edges[idx]
                        A[u, v] = -A[u, v]
                        A[v, u] = -A[v, u]

    if best_signs is None:
        # Ultimate fallback
        best_signs = np.ones(n_edges, dtype=int)
        for i, t in enumerate(edge_types):
            if t == 2:
                best_signs[i] = -1
                break
        if not is_valid_lift_fast(best_signs):
            best_signs = make_valid(best_signs, rng)

    return best_signs
