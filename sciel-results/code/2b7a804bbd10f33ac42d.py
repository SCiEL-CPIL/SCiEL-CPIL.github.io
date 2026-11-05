import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit

    def random_graph(n, p, seed):
        from random import Random
        rng = Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        rem = adj[:]  # remaining edges
        cliques = []

        # Precompute spectral ordering for vertex prioritization
        def compute_spectral_order(adj_list):
            """Compute eigenvector centrality to prioritize vertices."""
            A = np.zeros((n, n), dtype=np.float64)
            for u in range(n):
                m = adj_list[u]
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    A[u, v] = 1.0
                    m ^= lsb
            
            # Use power iteration for principal eigenvector
            if n == 0:
                return list(range(n))
            
            vec = np.ones(n, dtype=np.float64) / n
            for _ in range(50):
                new_vec = A @ vec
                norm = np.linalg.norm(new_vec)
                if norm == 0:
                    break
                new_vec /= norm
                if np.linalg.norm(new_vec - vec) < 1e-10:
                    break
                vec = new_vec
            
            # Sort by eigenvector centrality descending
            order = np.argsort(-vec)
            return [int(i) for i in order]

        spectral_order = compute_spectral_order(adj)
        spectral_rank = [0] * n
        for rank, v in enumerate(spectral_order):
            spectral_rank[v] = rank

        # Bron-Kerbosch with coloring to find a maximum clique in rem
        def find_max_clique():
            # Active vertices: those with at least one remaining edge
            active = 0
            for i in range(n):
                if rem[i]:
                    active |= (1 << i)

            if active == 0:
                return []

            best = []

            def color_sort(P):
                verts = []
                m = P
                while m:
                    lsb = m & -m
                    verts.append(lsb.bit_length() - 1)
                    m ^= lsb
                # Sort by degree in P descending, then by spectral rank (lower rank = higher centrality)
                verts.sort(key=lambda v: (-(rem[v] & P).bit_count(), spectral_rank[v]))
                colors = {}
                colored = []
                for v in verts:
                    used = set()
                    nm = rem[v] & P
                    while nm:
                        lsb = nm & -nm
                        u = lsb.bit_length() - 1
                        if u in colors:
                            used.add(colors[u])
                        nm ^= lsb
                    c = 1
                    while c in used:
                        c += 1
                    colors[v] = c
                    colored.append((c, v))
                colored.sort(key=lambda x: x[0])
                return colored

            def bk(R, P, X):
                nonlocal best
                if P == 0:
                    if len(R) > len(best):
                        best = list(R)
                    return
                colored = color_sort(P)
                for color, v in reversed(colored):
                    if len(R) + color <= len(best):
                        break
                    R.append(v)
                    bk(R, P & rem[v], X & rem[v])
                    R.pop()
                    P ^= (1 << v)
                    X |= (1 << v)

            bk([], active, 0)
            return best

        # Main loop: repeatedly find max clique, remove its edges
        while True:
            if time.time() > deadline:
                break

            clique = find_max_clique()
            if len(clique) < 2:
                break

            cliques.append(clique)

            # Remove all edges within clique from rem
            for i in range(len(clique)):
                u = clique[i]
                for j in range(i + 1, len(clique)):
                    v = clique[j]
                    rem[u] &= ~(1 << v)
                    rem[v] &= ~(1 << u)

            # Update spectral order if significant edges removed
            remaining_edge_count = sum(rem[i].bit_count() for i in range(n)) // 2
            if remaining_edge_count < 0.5 * (sum(adj[i].bit_count() for i in range(n)) // 2):
                spectral_order = compute_spectral_order(rem)
                for rank, v in enumerate(spectral_order):
                    spectral_rank[v] = rank

        # Fallback: cover any remaining edges as pairs
        for u in range(n):
            nm = rem[u]
            while nm:
                lsb = nm & -nm
                v = lsb.bit_length() - 1
                if u < v:
                    cliques.append([u, v])
                nm ^= lsb
                rem[u] &= ~(1 << v)
                rem[v] &= ~(1 << u)

        return cliques

    c1 = solve_instance(32, 0.6, 601)
    c2 = solve_instance(64, 0.55, 602)

    return [c1, c2]
