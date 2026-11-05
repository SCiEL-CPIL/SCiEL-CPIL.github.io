import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

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

    def spectral_seed_clique(adj, n, k=8):
        """Use the principal eigenvector of the adjacency matrix to seed a clique."""
        try:
            import numpy as np
            A = np.zeros((n, n), dtype=np.float64)
            for u in range(n):
                m = adj[u]
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    A[u, v] = 1.0
                    m ^= lsb
            # Power iteration for principal eigenvector
            v = np.ones(n, dtype=np.float64)
            v /= np.linalg.norm(v)
            for _ in range(200):
                v_new = A @ v
                nv = np.linalg.norm(v_new)
                if nv < 1e-12:
                    break
                v_new /= nv
                if np.abs(np.dot(v_new, v) - 1.0) < 1e-10:
                    v = v_new
                    break
                v = v_new
            # Take top-k vertices by eigenvector component
            idx = np.argsort(-v)[:k]
            seed = [int(i) for i in idx]
            # Greedily grow: try to make it a clique by removing non-adjacent vertices
            clique = []
            for vtx in seed:
                ok = True
                for c in clique:
                    if not (adj[c] >> vtx) & 1:
                        ok = False
                        break
                if ok:
                    clique.append(vtx)
            # Try to add more vertices from the top eigenvector list
            for vtx in idx[k:]:
                vtx = int(vtx)
                ok = True
                for c in clique:
                    if not (adj[c] >> vtx) & 1:
                        ok = False
                        break
                if ok:
                    clique.append(vtx)
            if len(clique) >= 2:
                return clique
        except Exception:
            pass
        return []

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        rem = adj[:]  # remaining edges
        cliques = []

        def find_max_clique():
            # Active vertices: those with at least one remaining edge
            active = 0
            for i in range(n):
                if rem[i]:
                    active |= (1 << i)

            if active == 0:
                return []

            best = []

            # Spectral seed for a strong initial clique
            seed_clique = spectral_seed_clique(rem, n)
            if len(seed_clique) > len(best):
                best = list(seed_clique)

            def color_sort(P):
                verts = []
                m = P
                while m:
                    lsb = m & -m
                    verts.append(lsb.bit_length() - 1)
                    m ^= lsb
                # Sort by degree in P descending
                verts.sort(key=lambda v: (rem[v] & P).bit_count(), reverse=True)
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
