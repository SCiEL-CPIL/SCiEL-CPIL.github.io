def construct():
    import time
    import random
    import sys

    start_time = time.time()
    DEADLINE = 145.0

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

    def spectral_order(adj, n, active_mask):
        """Use power iteration on adjacency to get eigenvector centrality ordering."""
        verts = []
        m = active_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
        if not verts:
            return []
        
        # Power iteration for top eigenvector of adjacency matrix restricted to active
        vec = {v: 1.0 for v in verts}
        for _ in range(50):
            new_vec = {}
            for v in verts:
                s = 0.0
                nb = adj[v] & active_mask
                while nb:
                    lsb = nb & -nb
                    u = lsb.bit_length() - 1
                    s += vec.get(u, 0.0)
                    nb ^= lsb
                new_vec[v] = s
            # Normalize
            norm = sum(x*x for x in new_vec.values()) ** 0.5
            if norm > 0:
                for v in new_vec:
                    new_vec[v] /= norm
            vec = new_vec
        
        # Sort by eigenvector centrality descending
        verts.sort(key=lambda v: vec.get(v, 0.0), reverse=True)
        return verts

    def max_clique_bk(adj, n, active_mask):
        """Bron-Kerbosch with coloring to find max clique in subgraph induced by active_mask."""
        if active_mask == 0:
            return []

        best = [0]
        best_clique = []

        verts = []
        m = active_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb

        if not verts:
            return []

        # Use spectral ordering for better branching
        verts = spectral_order(adj, n, active_mask)

        def color_sort(P_mask, verts_list):
            """Greedy coloring of vertices in P_mask, return list of (color, vertex) sorted by color."""
            if not verts_list:
                return []

            v_color = {}
            result = []
            for v in verts_list:
                if not (P_mask & (1 << v)):
                    continue
                used_colors = 0
                neigh = adj[v] & P_mask
                nm = neigh
                while nm:
                    lsb = nm & -nm
                    u = lsb.bit_length() - 1
                    if u in v_color:
                        used_colors |= (1 << v_color[u])
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

            # Get vertices in P for coloring
            p_verts = []
            pm = P
            while pm:
                lsb = pm & -pm
                idx = lsb.bit_length() - 1
                p_verts.append(idx)
                pm ^= lsb

            # Sort by degree in P for better coloring
            p_verts.sort(key=lambda v: (adj[v] & P).bit_count(), reverse=True)

            colored = color_sort(P, p_verts)
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

    def greedy_extend(clique, adj, n, active_mask):
        """Try to greedily extend a clique by adding vertices adjacent to all current members."""
        if not clique:
            return clique
        current = set(clique)
        # Find candidates adjacent to all current members
        candidate = active_mask
        for v in clique:
            candidate &= adj[v]
        # Remove already in clique
        for v in clique:
            candidate &= ~(1 << v)
        
        # Greedily add vertices
        changed = True
        while changed:
            changed = False
            cm = candidate
            while cm:
                lsb = cm & -cm
                v = lsb.bit_length() - 1
                # Check if v is adjacent to all in current
                ok = True
                for u in current:
                    if not (adj[v] & (1 << u)):
                        ok = False
                        break
                if ok:
                    current.add(v)
                    candidate &= ~(1 << v)
                    changed = True
                cm ^= lsb
        return list(current)

    def solve_instance(n, p, seed):
        adj = generate_graph(n, p, seed)
        rem = adj[:]
        cliques = []

        active = 0
        for i in range(n):
            if rem[i]:
                active |= (1 << i)

        while active and not time_up():
            # Find max clique in remaining graph
            clique = max_clique_bk(rem, n, active)

            if not clique:
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

            # Try to extend the clique greedily
            clique = greedy_extend(clique, rem, n, active)

            cliques.append(clique)

            # Remove clique edges from rem
            for u in clique:
                for v in clique:
                    if u != v:
                        rem[u] &= ~(1 << v)

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
