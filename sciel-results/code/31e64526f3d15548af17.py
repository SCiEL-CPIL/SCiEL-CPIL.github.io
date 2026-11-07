<<<<<<< SEARCH
def construct():
    t_start = time.time()
    deadline = t_start + 60.0 - DEADLINE_MARGIN

    # Build edges once
    edges = _build_edges()
    # Map vertex -> list of edge indices containing it
    vert_to_edges = {v: [] for v in range(1, N+1)}
    for i, e in enumerate(edges):
        for v in e:
            vert_to_edges[v].append(i)

    # Lagrangian relaxation: maximize |S| subject to for each edge e, not(all v in e in S)
    # Equivalently: for each edge e, sum_{v in e} x_v <= |e| - 1
    # Lagrangian: L(x, lambda) = sum x_v - sum_e lambda_e * max(0, sum_{v in e} x_v - (|e|-1))
    # Subgradient method on lambda, greedy primal on x given lambda.

    random.seed(42)
    best_set = set(range(N//2+1, N+1))  # seed solution
    best_size = len(best_set)

    # Greedy primal: given vertex weights w, greedily add vertices in order of w,
    # skipping if it would create a violation.
    def _greedy_primal(weights, max_time=None):
        """weights: dict or array indexed 1..N. Returns a set."""
        sel = set()
        # Track for each edge how many of its vertices are selected
        edge_count = np.zeros(len(edges), dtype=np.int8)
        # Process vertices in descending weight order
        order = sorted(range(1, N+1), key=lambda v: -weights[v])
        for v in order:
            if max_time is not None and time.time() > max_time:
                break
            # Check if adding v creates any violation
            # v is in edges vert_to_edges[v]
            conflict = False
            for ei in vert_to_edges[v]:
                if edge_count[ei] == len(edges[ei]) - 1:
                    # Adding v would make all vertices of this edge selected
                    conflict = True
                    break
            if not conflict:
                sel.add(v)
                for ei in vert_to_edges[v]:
                    edge_count[ei] += 1
        return sel

    # Subgradient iteration
    n_edges = len(edges)
    lam = np.zeros(n_edges)
    step_sizes = [10.0, 5.0, 2.0, 1.0, 0.5, 0.25, 0.1]
    
    for it in range(200):
        if time.time() > deadline:
            break
        # Compute vertex weights: w_v = 1 - sum_{e: v in e} lambda_e
        weights = np.ones(N+1)
        for ei in range(n_edges):
            if lam[ei] > 0:
                for v in edges[ei]:
                    weights[v] -= lam[ei]
        # Greedy primal
        sel = _greedy_primal(weights, max_time=deadline)
        if len(sel) > best_size:
            best_size = len(sel)
            best_set = sel.copy()
        
        # Subgradient update: gradient of L w.r.t. lam_e = max(0, sum_{v in e} x_v - (|e|-1))
        x = np.zeros(N+1)
        for v in sel:
            x[v] = 1.0
        grad = np.zeros(n_edges)
        for ei, e in enumerate(edges):
            s = sum(x[v] for v in e)
            cap = len(e) - 1
            if s > cap:
                grad[ei] = s - cap
        # Step size
        ss = step_sizes[min(it // 30, len(step_sizes)-1)]
        lam = np.maximum(0, lam + ss * grad)

    # Final polish: try to add any missing vertex that doesn't create a violation
    if time.time() < deadline:
        sel = set(best_set)
        edge_count = np.zeros(n_edges, dtype=np.int8)
        for ei, e in enumerate(edges):
            for v in e:
                if v in sel:
                    edge_count[ei] += 1
        # Try adding each non-selected vertex
        for v in range(1, N+1):
            if v in sel:
                continue
            if time.time() > deadline:
                break
            conflict = False
            for ei in vert_to_edges[v]:
                if edge_count[ei] == len(edges[ei]) - 1:
                    conflict = True
                    break
            if not conflict:
                sel.add(v)
                for ei in vert_to_edges[v]:
                    edge_count[ei] += 1
        if len(sel) > best_size:
            best_size = len(sel)
            best_set = sel.copy()

    # Convert to list of Python ints
    result = [int(v) for v in sorted(best_set)]
    return result
=======
def construct():
    t_start = time.time()
    deadline = t_start + 60.0 - DEADLINE_MARGIN

    # Build edges once
    edges = _build_edges()
    n_edges = len(edges)
    
    # Precompute edge sizes
    edge_sizes = np.array([len(e) for e in edges], dtype=np.int8)
    edge_caps = edge_sizes - 1  # max allowed selected vertices per edge
    
    # Map vertex -> list of edge indices containing it
    vert_to_edges = {v: [] for v in range(1, N+1)}
    for i, e in enumerate(edges):
        for v in e:
            vert_to_edges[v].append(i)
    
    # Precompute for each vertex, the list of (edge_idx, cap) pairs for fast conflict check
    vert_edge_info = {}
    for v in range(1, N+1):
        vert_edge_info[v] = [(ei, edge_caps[ei]) for ei in vert_to_edges[v]]

    random.seed(42)
    best_set = set(range(N//2+1, N+1))  # seed solution
    best_size = len(best_set)

    def _greedy_primal(weights, max_time=None, rng=None):
        """weights: array indexed 1..N. Returns a set. Optional rng for random tie-breaking."""
        sel = set()
        edge_count = np.zeros(n_edges, dtype=np.int8)
        
        # Process vertices in descending weight order; break ties randomly if rng provided
        if rng is not None:
            order = list(range(1, N+1))
            rng.shuffle(order)
            order.sort(key=lambda v: -weights[v], stable=True)
        else:
            order = sorted(range(1, N+1), key=lambda v: -weights[v])
        
        for v in order:
            if max_time is not None and time.time() > max_time:
                break
            conflict = False
            for ei, cap in vert_edge_info[v]:
                if edge_count[ei] == cap:
                    conflict = True
                    break
            if not conflict:
                sel.add(v)
                for ei, _ in vert_edge_info[v]:
                    edge_count[ei] += 1
        return sel

    def _polish(sel, max_time=None):
        """Iteratively add any vertex that doesn't create a violation. Returns improved set."""
        if max_time is not None and time.time() > max_time:
            return sel
        sel = set(sel)
        edge_count = np.zeros(n_edges, dtype=np.int8)
        for ei, e in enumerate(edges):
            for v in e:
                if v in sel:
                    edge_count[ei] += 1
        
        improved = True
        while improved:
            if max_time is not None and time.time() > max_time:
                break
            improved = False
            for v in range(1, N+1):
                if v in sel:
                    continue
                if max_time is not None and time.time() > max_time:
                    return sel
                conflict = False
                for ei, cap in vert_edge_info[v]:
                    if edge_count[ei] == cap:
                        conflict = True
                        break
                if not conflict:
                    sel.add(v)
                    for ei, _ in vert_edge_info[v]:
                        edge_count[ei] += 1
                    improved = True
        return sel

    def _local_search_swap(sel, max_time=None):
        """Try swapping: remove one vertex, then try to add two others. Returns improved set."""
        if max_time is not None and time.time() > max_time:
            return sel
        sel = set(sel)
        edge_count = np.zeros(n_edges, dtype=np.int8)
        for ei, e in enumerate(edges):
            for v in e:
                if v in sel:
                    edge_count[ei] += 1
        
        best_local = sel.copy()
        best_local_size = len(sel)
        
        for remove_v in range(1, N+1):
            if remove_v not in sel:
                continue
            if max_time is not None and time.time() > max_time:
                return best_local
            
            # Temporarily remove remove_v
            for ei, _ in vert_edge_info[remove_v]:
                edge_count[ei] -= 1
            
            # Try to add vertices that were previously blocked
            added = []
            for add_v in range(1, N+1):
                if add_v == remove_v or add_v in sel:
                    continue
                if max_time is not None and time.time() > max_time:
                    break
                conflict = False
                for ei, cap in vert_edge_info[add_v]:
                    if edge_count[ei] == cap:
                        conflict = True
                        break
                if not conflict:
                    sel.add(add_v)
                    added.append(add_v)
                    for ei, _ in vert_edge_info[add_v]:
                        edge_count[ei] += 1
            
            # If we added more than we removed, keep the improvement
            if len(added) > 1:
                if len(sel) > best_local_size:
                    best_local_size = len(sel)
                    best_local = sel.copy()
            else:
                # Revert additions
                for add_v in added:
                    sel.discard(add_v)
                    for ei, _ in vert_edge_info[add_v]:
                        edge_count[ei] -= 1
                # Restore remove_v
                sel.add(remove_v)
                for ei, _ in vert_edge_info[remove_v]:
                    edge_count[ei] += 1
        
        return best_local

    # Subgradient iteration with multi-start
    lam = np.zeros(n_edges)
    rng = random.Random(42)
    
    max_iter = 500
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Adaptive step size
        ss = max(0.01, 2.0 / (1.0 + 0.01 * it))
        
        # Compute vertex weights: w_v = 1 - sum_{e: v in e} lambda_e
        weights = np.ones(N+1)
        for ei in range(n_edges):
            if lam[ei] > 0:
                for v in edges[ei]:
                    weights[v] -= lam[ei]
        
        # Greedy primal with random tie-breaking for diversity
        sel = _greedy_primal(weights, max_time=deadline, rng=rng)
        
        # Polish
        sel = _polish(sel, max_time=deadline)
        
        if len(sel) > best_size:
            best_size = len(sel)
            best_set = sel.copy()
        
        # Subgradient update
        x = np.zeros(N+1)
        for v in sel:
            x[v] = 1.0
        grad = np.zeros(n_edges)
        for ei, e in enumerate(edges):
            s = 0
            for v in e:
                s += x[v]
            if s > edge_caps[ei]:
                grad[ei] = s - edge_caps[ei]
        
        lam = np.maximum(0, lam + ss * grad)
        
        # Occasionally do local search on the best set
        if it % 50 == 0 and best_size > 0 and time.time() < deadline:
            polished = _local_search_swap(best_set, max_time=deadline)
            if len(polished) > best_size:
                best_size = len(polished)
                best_set = polished.copy()

    # Final comprehensive polish
    if time.time() < deadline:
        final = _polish(best_set, max_time=deadline)
        if len(final) > best_size:
            best_size = len(final)
            best_set = final.copy()
    
    # Final local search if time permits
    if time.time() < deadline - 2.0:
        final = _local_search_swap(best_set, max_time=deadline)
        if len(final) > best_size:
            best_size = len(final)
            best_set = final.copy()

    # Convert to list of Python ints
    result = [int(v) for v in sorted(best_set)]
    return result
>>>>>>> REPLACE
