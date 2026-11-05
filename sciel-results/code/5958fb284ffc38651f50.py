import time
import numpy as np
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def random_graph(n, p, seed):
        rng = Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def solve_instance(n, p, seed, time_limit):
        adj = random_graph(n, p, seed)
        rem = adj[:]
        cliques = []
        total_edges = sum(adj[i].bit_count() for i in range(n)) // 2
        if total_edges == 0:
            return cliques

        # Compute spectral order
        def compute_spectral_order(adj_list):
            A = np.zeros((n, n), dtype=np.float64)
            for u in range(n):
                m = adj_list[u]
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    A[u, v] = 1.0
                    m ^= lsb
            if n == 0:
                return list(range(n))
            vec = np.ones(n, dtype=np.float64) / n
            for _ in range(30):
                new_vec = A @ vec
                norm = np.linalg.norm(new_vec)
                if norm == 0:
                    break
                new_vec /= norm
                if np.linalg.norm(new_vec - vec) < 1e-10:
                    break
                vec = new_vec
            order = np.argsort(-vec)
            return [int(i) for i in order]

        spectral_order = compute_spectral_order(adj)
        spectral_rank = [0] * n
        for rank, v in enumerate(spectral_order):
            spectral_rank[v] = rank

        def find_max_clique():
            active = 0
            for i in range(n):
                if rem[i]:
                    active |= (1 << i)
            if active == 0:
                return []

            best = []

            # Greedy initial clique for better pruning bound
            def greedy_clique(start_mask):
                clique = []
                P = start_mask
                while P:
                    # Pick vertex with max degree in P
                    best_v = -1
                    best_deg = -1
                    m = P
                    while m:
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        deg = (rem[v] & P).bit_count()
                        if deg > best_deg:
                            best_deg = deg
                            best_v = v
                        m ^= lsb
                    if best_v < 0:
                        break
                    clique.append(best_v)
                    P &= rem[best_v]
                    P &= ~(1 << best_v)
                return clique

            greedy = greedy_clique(active)
            if len(greedy) > len(best):
                best = list(greedy)

            def color_sort(P):
                verts = []
                m = P
                while m:
                    lsb = m & -m
                    verts.append(lsb.bit_length() - 1)
                    m ^= lsb
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
                    P &= ~(1 << v)
                    X |= (1 << v)

            bk([], active, 0)
            return best

        while time.time() < time_limit:
            has_edges = False
            for i in range(n):
                if rem[i]:
                    has_edges = True
                    break
            if not has_edges:
                break

            clique = find_max_clique()
            if len(clique) < 2:
                break

            cliques.append(clique)

            for i in range(len(clique)):
                u = clique[i]
                for j in range(i + 1, len(clique)):
                    v = clique[j]
                    rem[u] &= ~(1 << v)
                    rem[v] &= ~(1 << u)

            remaining = sum(rem[i].bit_count() for i in range(n)) // 2
            if remaining < 0.5 * total_edges:
                spectral_order = compute_spectral_order(rem)
                for rank, v in enumerate(spectral_order):
                    spectral_rank[v] = rank

        # Fallback
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

    # Allocate time: give more to first instance since it's smaller
    t1 = time.time()
    c1 = solve_instance(32, 0.6, 601, time_limit=start_time + 55.0)
    t2 = time.time()
    # Use remaining time for second instance
    remaining_time = deadline - t2
    if remaining_time < 10:
        remaining_time = 10
    c2 = solve_instance(64, 0.55, 602, time_limit=t2 + remaining_time)

    return [c1, c2]
