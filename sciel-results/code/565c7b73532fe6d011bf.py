import time
import random as _random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    def random_graph(n, p, seed):
        rng = _random.Random(seed)
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

        def find_max_clique():
            # Active vertices: those with at least one remaining edge
            active = 0
            for i in range(n):
                if rem[i]:
                    active |= (1 << i)

            if active == 0:
                return []

            best = []

            # Greedy seed: repeatedly pick highest-degree vertex, grow clique
            def greedy_seed():
                nonlocal best
                P = active
                R = []
                while P:
                    # Pick vertex with max degree in P
                    max_deg = -1
                    max_v = -1
                    m = P
                    while m:
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        d = (rem[v] & P).bit_count()
                        if d > max_deg:
                            max_deg = d
                            max_v = v
                        m ^= lsb
                    if max_v < 0:
                        break
                    R.append(max_v)
                    P &= rem[max_v]
                    P ^= (1 << max_v)
                if len(R) > len(best):
                    best = list(R)
                return R

            greedy_seed()

            # Bron-Kerbosch with coloring for maximum clique
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
