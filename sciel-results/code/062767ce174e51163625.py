def sign_edges(graph, budget):
    import numpy as np
    import time
    from scipy.linalg import eigh

    start_time = time.time()
    DEADLINE = 82  # safety margin before 90s

    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m

    edge_u = np.array([u for u, v in edges], dtype=int)
    edge_v = np.array([v for u, v in edges], dtype=int)

    # Classify edges
    # Bottom cycle: both < m
    # Top cycle: both >= m
    # Matching: one < m, one >= m
    is_bottom = (edge_u < m) & (edge_v < m)
    is_top = (edge_u >= m) & (edge_v >= m)
    is_matching = ~(is_bottom | is_top)

    # For prism graph, there are exactly 3 types of cycles:
    # 1. Bottom m-cycle
    # 2. Top m-cycle
    # 3. m "ladder" cycles (each goes through one matching edge on each side)
    # Actually, the fundamental cycles depend on spanning tree choice.
    # Let's use a specific spanning tree: all matching edges + all but one bottom edge + all but one top edge
    # No wait, simpler: the signed adjacency matrix eigenvalues for a prism graph lift
    # can be computed analytically.

    # The signed adjacency matrix A is 2m x 2m.
    # For a prism graph, the eigenvalues of the lift's adjacency matrix are related to
    # the eigenvalues of certain 2x2 or m x m matrices.

    # Actually, let's think about this differently. The spectral radius of the signed
    # adjacency matrix of the BASE graph (2m x 2m) is what we're minimizing.
    # Wait, re-reading: "spectral radius of the signed adjacency matrix of the base graph"
    # The base graph has 2m vertices. The signed adjacency matrix is 2m x 2m.

    # For the prism graph with signs, the adjacency matrix has a specific structure.
    # Let's index vertices 0..m-1 (bottom) and m..2m-1 (top).
    # Bottom cycle: (0,1), (1,2), ..., (m-1, 0)
    # Top cycle: (m, m+1), (m+1, m+2), ..., (2m-1, m)
    # Matching: (i, i+m) for i=0..m-1

    # The signed adjacency matrix A has:
    # A[i, j] = sign of edge (i,j) for adjacent vertices

    # Key insight: The eigenvalues of A can be found by exploiting the structure.
    # A is a circulant-like matrix if signs are periodic, but in general it's just a
    # sparse matrix. For m <= 9, 2m <= 18, so eigenvalue computation is fast.

    # Strategy: Use the fact that the spectrum depends on the "frustration" of cycles.
    # For a 2-lift to be connected, we need at least one cycle with odd number of -1 signs.

    # Let's enumerate all 2^(number of independent cycles) sign patterns up to switching.
    # The number of independent cycles in a prism graph: E - V + 1 = 3m - 2m + 1 = m + 1
    # So there are 2^(m+1) switching classes. For m=9, that's 2^10 = 1024.
    # We can enumerate all switching classes and find the one with minimum spectral radius.

    # To enumerate switching classes, we fix a reference and vary signs on non-tree edges.
    # Build a spanning tree:
    # Use matching edges (i, i+m) for all i, plus bottom edges (0,1), (1,2), ..., (m-2, m-1)
    # That's m + (m-1) = 2m-1 edges. Non-tree edges: 3m - (2m-1) = m+1 edges.
    # Non-tree edges are: bottom edge (m-1, 0) and all top cycle edges.

    # Actually, let me just directly enumerate by fixing the first matching edge sign to +1
    # (switching equivalence) and varying the other m non-tree edges.

    # Simpler approach: just enumerate all 2^(m+1) combinations of signs on a set of
    # m+1 "independent" edges, keeping the rest determined by the spanning tree.

    # Let me identify the edges explicitly.
    # Assume edges are ordered as: bottom cycle, top cycle, matching (or some order).
    # Let me just work with the edge list as given.

    # Build spanning tree via BFS
    adj_list = [[] for _ in range(n_vertices)]
    for i in range(n_edges):
        u, v = edge_u[i], edge_v[i]
        adj_list[u].append((v, i))
        adj_list[v].append((u, i))

    parent = [-1] * n_vertices
    parent_edge = [-1] * n_vertices
    visited = [False] * n_vertices
    visited[0] = True
    queue = [0]
    qi = 0
    while qi < len(queue):
        node = queue[qi]
        qi += 1
        for nb, ei in adj_list[node]:
            if not visited[nb]:
                visited[nb] = True
                parent[nb] = node
                parent_edge[nb] = ei
                queue.append(nb)

    tree_edges = set(parent_edge[1:])
    non_tree_edges = [i for i in range(n_edges) if i not in tree_edges]
    n_non_tree = len(non_tree_edges)  # should be m+1

    def build_A_from_non_tree_signs(non_tree_signs):
        """Build signed adjacency matrix given signs on non-tree edges.
        Tree edge signs are determined by switching: we fix vertex 0's gauge to +1,
        so tree edge (parent[v], v) has sign = +1 * sign[v] / sign[parent[v]].
        Actually, for a 2-lift, the sign on tree edge is determined by the requirement
        that the product around any cycle is consistent. Let me think again.

        In a 2-lift, each edge e has a sign s_e in {+1, -1}. The lift is defined by:
        - If s_e = +1, vertex (u,0) connects to (v,0) and (u,1) connects to (v,1)
        - If s_e = -1, vertex (u,0) connects to (v,1) and (u,1) connects to (v,0)

        The signed adjacency matrix A of the base graph has A[u,v] = s_e for edge (u,v).

        The eigenvalues of A give the "new" eigenvalues of the lift.

        For switching: if we multiply all signs incident to vertex v by -1, the lift
        is isomorphic (just relabel (v,0) <-> (v,1)). So the spectrum is invariant
        under switching.

        To enumerate switching classes: fix sign of all tree edges to +1 (by choosing
        appropriate vertex gauges), then vary signs of non-tree edges.
        """
        # Determine tree edge signs from vertex gauges
        # Fix gauge[0] = +1
        # For tree edge (parent[v], v): sign = gauge[parent[v]] * gauge[v]
        # We want all tree edge signs to be +1, so gauge[v] = gauge[parent[v]]
        # Thus all gauges are +1, and tree edge signs are all +1.
        # Then non-tree edge signs are free parameters.

        A = np.zeros((n_vertices, n_vertices))
        # Set tree edges to +1
        for v in range(1, n_vertices):
            pe = parent_edge[v]
            u = parent[v]
            A[u, v] = 1
            A[v, u] = 1
        # Set non-tree edges
        for k, ei in enumerate(non_tree_edges):
            s = non_tree_signs[k]
            u = edge_u[ei]
            v = edge_v[ei]
            A[u, v] = s
            A[v, u] = s
        return A

    def spectral_radius(A):
        try:
            vals = eigh(A, eigvals_only=True)
            return np.max(np.abs(vals))
        except:
            return np.max(np.abs(np.linalg.eigvalsh(A)))

    def is_connected_lift(A):
        """Check if the 2-lift is connected.
        The lift is connected iff the signed graph is not balanced,
        i.e., there exists a cycle with product of signs = -1.
        """
        # Check all fundamental cycles
        for ei in non_tree_edges:
            u, v = edge_u[ei], edge_v[ei]
            # Find path from u to v in tree
            path_u = []
            path_v = []
            cu = u
            while cu != 0:
                path_u.append(cu)
                cu = parent[cu]
            path_u.append(0)
            cv = v
            while cv != 0:
                path_v.append(cv)
                cv = parent[cv]
            path_v.append(0)
            # LCA
            set_u = set(path_u)
            lca = 0
            for node in path_v:
                if node in set_u:
                    lca = node
                    break
            # Product of signs around cycle
            prod = A[edge_u[ei], edge_v[ei]]  # the non-tree edge
            # Multiply by tree edge signs (all +1 in our gauge)
            # Since all tree edges are +1, the product is just the non-tree edge sign
            # Wait, that's not right. The cycle includes tree edges whose signs are +1
            # in our gauge, so the product is just s_ei.
            # Actually no - the product around the cycle is the product of all edge signs.
            # In our gauge, tree edges have sign +1, so product = s_ei.
            if prod == -1:
                return True
        return False

    # Enumerate all 2^(m+1) switching classes
    best_radius = float('inf')
    best_non_tree_signs = None

    n_combinations = 1 << n_non_tree
    for mask in range(n_combinations):
        if time.time() - start_time > DEADLINE - 20:
            break

        non_tree_signs = np.ones(n_non_tree, dtype=int)
        for k in range(n_non_tree):
            if mask & (1 << k):
                non_tree_signs[k] = -1

        A = build_A_from_non_tree_signs(non_tree_signs)

        # Check connectivity
        if not is_connected_lift(A):
            continue

        radius = spectral_radius(A)
        if radius < best_radius:
            best_radius = radius
            best_non_tree_signs = non_tree_signs.copy()

    # Convert best non-tree signs to full sign vector
    if best_non_tree_signs is not None:
        signs = np.ones(n_edges, dtype=int)
        # Tree edges are +1
        for v in range(1, n_vertices):
            pe = parent_edge[v]
            signs[pe] = 1
        # Non-tree edges
        for k, ei in enumerate(non_tree_edges):
            signs[ei] = best_non_tree_signs[k]
    else:
        # Fallback: make a valid sign assignment
        signs = np.ones(n_edges, dtype=int)
        for ei in non_tree_edges:
            signs[ei] = -1
            break
        # Ensure validity
        if not is_connected_lift(build_A_from_non_tree_signs(np.array([-1]*n_non_tree))):
            signs = np.ones(n_edges, dtype=int)
            for i in range(n_edges):
                u, v = edge_u[i], edge_v[i]
                if not (u < m and v < m) and not (u >= m and v >= m):
                    signs[i] = -1
                    break

    # Now do local search to potentially improve
    A = np.zeros((n_vertices, n_vertices))
    for i in range(n_edges):
        s = signs[i]
        u = edge_u[i]
        v = edge_v[i]
        A[u, v] = s
        A[v, u] = s

    current_radius = spectral_radius(A)

    def local_search(signs, A, current_radius, max_time):
        improved = True
        while improved and (time.time() - start_time < max_time):
            improved = False
            for i in range(n_edges):
                if time.time() - start_time > max_time:
                    break
                old = signs[i]
                u = edge_u[i]
                v = edge_v[i]
                signs[i] = -old
                A[u, v] = -A[u, v]
                A[v, u] = -A[v, u]

                if not is_connected_lift(A):
                    signs[i] = old
                    A[u, v] = -A[u, v]
                    A[v, u] = -A[v, u]
                    continue

                new_radius = spectral_radius(A)
                if new_radius < current_radius - 1e-12:
                    current_radius = new_radius
                    improved = True
                else:
                    signs[i] = old
                    A[u, v] = -A[u, v]
                    A[v, u] = -A[v, u]
        return signs, current_radius

    # Local search from the best enumerated solution
    if time.time() - start_time < DEADLINE - 5:
        signs, current_radius = local_search(signs.copy(), A.copy(), current_radius, DEADLINE - 2)

    # Additional: try a few random perturbations + local search
    rng = np.random.default_rng(42)
    for _ in range(10):
        if time.time() - start_time > DEADLINE - 5:
            break
        trial_signs = signs.copy()
        for _ in range(rng.integers(1, 4)):
            idx = rng.integers(0, n_edges)
            trial_signs[idx] *= -1
        trial_A = np.zeros((n_vertices, n_vertices))
        for i in range(n_edges):
            s = trial_signs[i]
            u = edge_u[i]
            v = edge_v[i]
            trial_A[u, v] = s
            trial_A[v, u] = s
        if not is_connected_lift(trial_A):
            continue
        trial_signs, trial_radius = local_search(trial_signs, trial_A, spectral_radius(trial_A), DEADLINE - 2)
        if trial_radius < current_radius:
            current_radius = trial_radius
            signs = trial_signs
            A = trial_A

    return signs
