def construct():
    import time
    import random
    import sys

    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    def time_up():
        return (time.time() - start_time) > DEADLINE

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def max_clique_bk(adj, n, active_mask):
        """Bron-Kerbosch with coloring to find max clique in subgraph induced by active_mask."""
        if active_mask == 0:
            return []

        best = [0]  # best clique size found
        best_clique = []

        # Get list of active vertices
        verts = []
        m = active_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb

        if not verts:
            return []

        # Sort by degree in active_mask descending
        verts.sort(key=lambda v: (adj[v] & active_mask).bit_count(), reverse=True)

        def color_sort(P_mask):
            """Greedy coloring of vertices in P_mask, return list of (color, vertex) sorted by color."""
            pm = P_mask
            verts_p = []
            while pm:
                lsb = pm & -pm
                idx = lsb.bit_length() - 1
                verts_p.append(idx)
                pm ^= lsb

            if not verts_p:
                return []

            # Sort by degree in P_mask
            verts_p.sort(key=lambda v: (adj[v] & P_mask).bit_count(), reverse=True)

            # Use a list indexed by vertex for color storage (faster than dict)
            v_color = [0] * n
            result = []
            for v in verts_p:
                # Collect used colors via bitmask
                used_colors = 0
                neigh = adj[v] & P_mask
                nm = neigh
                while nm:
                    lsb = nm & -nm
                    u = lsb.bit_length() - 1
                    cu = v_color[u]
                    if cu:
                        used_colors |= (1 << cu)
                    nm ^= lsb
                c = 1
                while (used_colors >> c) & 1:
                    c += 1
                v_color[v] = c
                result.append((c, v))

            result.sort(key=lambda x: x[0])
            return result

        def bk(R, P, X):
            nonlocal best, best_clique

            if P == 0 and X == 0:
                if len(R) > best[0]:
                    best[0] = len(R)
                    best_clique = list(R)
                return

            if time_up():
                return

            colored = color_sort(P)
            for color, v in reversed(colored):
                if len(R) + color <= best[0]:
                    break

                R.append(v)
                bk(R, P & adj[v], X & adj[v])
                R.pop()
                P ^= (1 << v)
                X |= (1 << v)

                if time_up():
                    return

        bk([], active_mask, 0)
        return best_clique

    def find_large_clique_heuristic(adj, n, active_mask):
        """
        Find a large clique using a heuristic based on eigenvector centrality.
        This is faster than exact BK and can find good cliques in dense graphs.
        """
        if active_mask == 0:
            return []
        
        # Get active vertices
        verts = []
        m = active_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
        
        if not verts:
            return []
        
        # Compute degrees in active subgraph
        degrees = {}
        for v in verts:
            degrees[v] = (adj[v] & active_mask).bit_count()
        
        # Iterative refinement: start with high-degree vertices and expand
        # Use power iteration for approximate eigenvector centrality
        n_active = len(verts)
        if n_active < 2:
            return list(verts)
        
        # Initialize with degree
        centrality = {v: float(degrees[v]) for v in verts}
        
        # A few iterations of power iteration
        for _ in range(3):
            new_centrality = {}
            for v in verts:
                s = 0.0
                neigh = adj[v] & active_mask
                nm = neigh
                while nm:
                    lsb = nm & -nm
                    u = lsb.bit_length() - 1
                    s += centrality.get(u, 0.0)
                    nm ^= lsb
                new_centrality[v] = s
            # Normalize
            max_val = max(new_centrality.values()) if new_centrality else 1.0
            if max_val > 0:
                for v in verts:
                    new_centrality[v] /= max_val
            centrality = new_centrality
        
        # Sort vertices by centrality
        verts_sorted = sorted(verts, key=lambda v: centrality[v], reverse=True)
        
        # Greedy clique construction: start with top vertex, add neighbors that maintain clique property
        if not verts_sorted:
            return []
        
        clique = [verts_sorted[0]]
        clique_set = {clique[0]}
        
        for v in verts_sorted[1:]:
            if v in clique_set:
                continue
            # Check if v is connected to all vertices in current clique
            is_connected = True
            for u in clique:
                if not (adj[v] & (1 << u)):
                    is_connected = False
                    break
            if is_connected:
                clique.append(v)
                clique_set.add(v)
        
        return clique

    def solve_instance(n, p, seed):
        adj = generate_graph(n, p, seed)
        rem = adj[:]  # remaining edges
        cliques = []

        # Compute initial active mask
        active = 0
        for i in range(n):
            if rem[i]:
                active |= (1 << i)

        while active and not time_up():
            # First try heuristic for speed, then refine with BK if time permits
            clique = find_large_clique_heuristic(rem, n, active)
            
            # If heuristic gives a small clique, try BK for potentially larger one
            if clique and len(clique) < 4:
                bk_clique = max_clique_bk(rem, n, active)
                if bk_clique and len(bk_clique) > len(clique):
                    clique = bk_clique
            elif not clique:
                clique = max_clique_bk(rem, n, active)

            if not clique:
                # Shouldn't happen if active has edges, but fallback
                # Find any edge
                found = False
                m = active
                while m and not found:
                    lsb = m & -m
                    u = lsb.bit_length() - 1
                    nb = rem[u] & active
                    if nb:
                        lsb2 = nb & -nb
                        v = lsb2.bit_length() - 1
                        clique = [u, v]
                        found = True
                    m ^= lsb
                if not found:
                    break

            cliques.append(clique)

            # Remove clique edges from rem
            if len(clique) >= 2:
                # Build mask of all vertices in clique
                clique_mask = 0
                for v in clique:
                    clique_mask |= (1 << v)
                for u in clique:
                    rem[u] &= ~clique_mask
                # Also clear self-loops if any
                for u in clique:
                    rem[u] &= ~(1 << u)

            # Update active mask
            active = 0
            for i in range(n):
                if rem[i]:
                    active |= (1 << i)

        # Fallback: cover remaining edges as pairs
        if active:
            for u in range(n):
                nb = rem[u]
                while nb:
                    lsb = nb & -nb
                    v = lsb.bit_length() - 1
                    if u < v:
                        cliques.append([u, v])
                        rem[u] &= ~(1 << v)
                        rem[v] &= ~(1 << u)
                    nb ^= lsb

        return cliques

    c1 = solve_instance(32, 0.6, 601)
    c2 = solve_instance(64, 0.55, 602)

    return [c1, c2]
