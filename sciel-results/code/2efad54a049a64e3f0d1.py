"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110.0  # Safety margin before 120s limit

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

    def compute_spectral_order(adj, n):
        """Use numpy power iteration for eigenvector centrality."""
        A = np.zeros((n, n), dtype=np.float64)
        for u in range(n):
            m = adj[u]
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                A[u, v] = 1.0
                m ^= lsb

        if n == 0:
            return list(range(n))

        # Initialize with degree
        vec = A.sum(axis=1).astype(np.float64)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        else:
            vec = np.ones(n, dtype=np.float64) / n

        # Power iteration
        for _ in range(30):
            new_vec = A @ vec
            norm = np.linalg.norm(new_vec)
            if norm == 0:
                break
            new_vec /= norm
            if np.linalg.norm(new_vec - vec) < 1e-12:
                break
            vec = new_vec

        order = np.argsort(-vec)
        return [int(i) for i in order]

    def max_clique_bk(rem, n, active_mask, spectral_rank):
        """Bron-Kerbosch with coloring to find max clique in subgraph induced by active_mask."""
        if active_mask == 0:
            return []

        best_size = [0]
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

        # Sort by spectral rank (lower rank = higher centrality)
        verts.sort(key=lambda v: spectral_rank[v])

        def color_sort(P_mask):
            """Greedy coloring of vertices in P_mask, return list of (color, vertex)."""
            pm = P_mask
            verts_p = []
            while pm:
                lsb = pm & -pm
                idx = lsb.bit_length() - 1
                verts_p.append(idx)
                pm ^= lsb

            if not verts_p:
                return []

            # Sort by degree in P_mask descending, then by spectral rank
            verts_p.sort(key=lambda v: (-(rem[v] & P_mask).bit_count(), spectral_rank[v]))

            v_color = {}
            result = []
            for v in verts_p:
                used = set()
                neigh = rem[v] & P_mask
                nm = neigh
                while nm:
                    lsb = nm & -nm
                    u = lsb.bit_length() - 1
                    if u in v_color:
                        used.add(v_color[u])
                    nm ^= lsb
                c = 1
                while c in used:
                    c += 1
                v_color[v] = c
                result.append((c, v))

            result.sort(key=lambda x: x[0])
            return result

        def bk(R, P, X):
            nonlocal best_size, best_clique

            if P == 0 and X == 0:
                if len(R) > best_size[0]:
                    best_size[0] = len(R)
                    best_clique = list(R)
                return

            if time_up():
                return

            colored = color_sort(P)
            for color, v in reversed(colored):
                if len(R) + color <= best_size[0]:
                    break

                R.append(v)
                bk(R, P & rem[v], X & rem[v])
                R.pop()
                P ^= (1 << v)
                X |= (1 << v)

                if time_up():
                    return

        bk([], active_mask, 0)
        return best_clique

    def solve_instance(n, p, seed):
        adj = generate_graph(n, p, seed)
        rem = adj[:]  # remaining edges as bitmasks
        cliques = []

        # Compute initial spectral ordering
        spectral_order = compute_spectral_order(adj, n)
        spectral_rank = [0] * n
        for rank, v in enumerate(spectral_order):
            spectral_rank[v] = rank

        # Track initial edge count for dynamic re-computation
        initial_edge_count = sum(adj[i].bit_count() for i in range(n)) // 2

        # Compute initial active mask
        active = 0
        for i in range(n):
            if rem[i]:
                active |= (1 << i)

        while active and not time_up():
            # Find max clique in remaining graph
            clique = max_clique_bk(rem, n, active, spectral_rank)

            if not clique or len(clique) < 2:
                # Fallback: find any edge
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
            for u in clique:
                for v in clique:
                    if u != v:
                        rem[u] &= ~(1 << v)

            # Update active mask
            active = 0
            for i in range(n):
                if rem[i]:
                    active |= (1 << i)

            # Dynamic re-computation: if significant edges removed, update spectral order
            remaining_edge_count = sum(rem[i].bit_count() for i in range(n)) // 2
            if initial_edge_count > 0 and remaining_edge_count < 0.4 * initial_edge_count:
                if not time_up():
                    spectral_order = compute_spectral_order(rem, n)
                    for rank, v in enumerate(spectral_order):
                        spectral_rank[v] = rank

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
# EVOLVE-BLOCK-END
