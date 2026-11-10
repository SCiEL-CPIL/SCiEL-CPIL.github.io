import time
import random
import sys

sys.setrecursionlimit(100000)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s/150s kill

    n = 18
    all_mask = (1 << n) - 1

    # ---------- helpers ----------

    def is_k4_free(adj):
        # adj: list of n bitmasks
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    continue
                common = adj[i] & adj[j] & all_mask & ~((1 << i) | (1 << j))
                # for each pair (k, l) in common with k < l, if edge (k,l) exists -> K4
                cm = common
                while cm:
                    lsb = cm & -cm
                    k = lsb.bit_length() - 1
                    rest = cm & (all_mask ^ ((1 << k) | lsb))
                    # need an l in rest such that adj[k] has l
                    if rest & adj[k]:
                        return False
                    cm ^= lsb
        return True

    def get_triangles(adj):
        tris = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    continue
                common = adj[i] & adj[j] & all_mask & ~((1 << i) | (1 << j))
                # only k > j to avoid dup
                cm = common & (all_mask ^ ((1 << (j + 1)) | ((1 << (j + 1)) - 1)))
                while cm:
                    lsb = cm & -cm
                    k = lsb.bit_length() - 1
                    tris.append((i, j, k))
                    cm ^= lsb
        return tris

    def tri_masks_of(tris):
        return [(1 << a) | (1 << b) | (1 << c) for a, b, c in tris]

    def max_triangle_free(adj, tris):
        """Exact: min vertex cover of triangle hypergraph via B&B.
        Returns n - min_cover."""
        if not tris:
            return n
        masks = tri_masks_of(tris)
        # order triangles: those with fewer "free" vertices first (heuristic)
        # Build incidence: for each triangle, list of its 3 vertices
        # We'll branch on a triangle, trying to exclude one of its vertices.

        # Precompute, for each vertex, which triangles contain it (index list)
        vert_to_tris = [[] for _ in range(n)]
        for ti, (a, b, c) in enumerate(tris):
            vert_to_tris[a].append(ti)
            vert_to_tris[b].append(ti)
            vert_to_tris[c].append(ti)

        # We maintain a list of "active" triangle indices.
        # B&B state: active_tris (list), excluded set (int bitmask), depth.
        # Prune: if len(active_tris)==0 -> covered.
        # Lower bound on additional excludes: ceil(|active_tris| / max_tris_per_vertex)
        #   where max_tris_per_vertex = max over v in remaining of count of active tris containing v.
        # Upper bound: current excluded count + (a greedy cover of remaining).

        best_cover = [n]  # mutable
        t0 = time.time()

        # Greedy initial upper bound: repeatedly pick vertex in most active tris
        def greedy_cover(active_tris):
            # active_tris: list of triangle index tuples (a,b,c)
            if not active_tris:
                return 0
            remaining = list(active_tris)
            chosen = 0
            while remaining:
                # count freq
                freq = {}
                for (a, b, c) in remaining:
                    for v in (a, b, c):
                        freq[v] = freq.get(v, 0) + 1
                # pick max
                v = max(freq, key=lambda x: (freq[x], -x))
                chosen += 1
                remaining = [t for t in remaining if v not in t]
            return chosen

        best_cover[0] = greedy_cover(tris)

        # B&B
        def rec(active_tris, excluded_mask, excluded_count):
            nonlocal best_cover
            if time.time() - t0 > 8.0:
                return
            if not active_tris:
                if excluded_count < best_cover[0]:
                    best_cover[0] = excluded_count
                return
            if excluded_count >= best_cover[0]:
                return
            # lower bound: each exclude can cover at most (max freq) tris
            freq = {}
            for (a, b, c) in active_tris:
                for v in (a, b, c):
                    if (excluded_mask >> v) & 1:
                        continue
                    freq[v] = freq.get(v, 0) + 1
            if not freq:
                return
            maxf = max(freq.values())
            lb = (len(active_tris) + maxf - 1) // maxf
            if excluded_count + lb >= best_cover[0]:
                return

            # pick a triangle to branch on: one with fewest "live" (not excluded) vertices
            # and among those, prefer triangles with high-frequency vertices
            best_tri = None
            best_key = None
            for t in active_tris:
                a, b, c = t
                live = []
                for v in (a, b, c):
                    if not ((excluded_mask >> v) & 1):
                        live.append(v)
                if not live:
                    continue
                # key: (len(live), -sum freq)
                s = sum(freq.get(v, 0) for v in live)
                key = (len(live), -s)
                if best_key is None or key < best_key:
                    best_key = key
                    best_tri = t
            if best_tri is None:
                # all tris already hit
                if excluded_count < best_cover[0]:
                    best_cover[0] = excluded_count
                return
            a, b, c = best_tri
            # try excluding each live vertex of this triangle
            for v in (a, b, c):
                if (excluded_mask >> v) & 1:
                    continue
                new_mask = excluded_mask | (1 << v)
                # remove all tris containing v
                new_active = [t for t in active_tris if v not in t]
                rec(new_active, new_mask, excluded_count + 1)

        rec(tris, 0, 0)
        return n - best_cover[0]

    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def adj_from_edges(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    # ---------- initial candidates ----------

    candidates = []

    # 1) K6,6,6 (complete tripartite)
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    seed_edges = [[a, b] for i in range(3) for j in range(i + 1, 3)
                  for a in parts[i] for b in parts[j]]
    candidates.append(seed_edges)

    # 2) Random graphs with K4 removal
    def random_k4_free():
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.45:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        # remove K4s greedily
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] >> j) & 1:
                        continue
                    common = adj[i] & adj[j] & all_mask & ~((1 << i) | (1 << j))
                    cm = common
                    while cm:
                        lsb = cm & -cm
                        k = lsb.bit_length() - 1
                        rest = cm & (all_mask ^ ((1 << k) | lsb))
                        if rest & adj[k]:
                            # K4 on i,j,k,l where l in rest&adj[k]
                            lsb2 = (rest & adj[k]) & -(rest & adj[k])
                            l = lsb2.bit_length() - 1
                            adj[i] &= ~(1 << l)
                            adj[l] &= ~(1 << i)
                            changed = True
                            break
                        cm ^= lsb
                    if changed:
                        break
                if changed:
                    break
            if changed:
                break
        return adj

    # ---------- local search ----------

    def try_local_search(adj, max_iters=200):
        # single-edge flips that keep K4-free and reduce score
        tris = get_triangles(adj)
        s = max_triangle_free(adj, tris)
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            improved = False
            # try removing a random edge
            for _ in range(30):
                i = random.randrange(n)
                j = random.randrange(n)
                if i == j or not ((adj[i] >> j) & 1):
                    continue
                # remove edge
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
                if not is_k4_free(adj):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    continue
                tris2 = get_triangles(adj)
                s2 = max_triangle_free(adj, tris2)
                if s2 < s:
                    s = s2
                    improved = True
                    break
                else:
                    # revert
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
            if improved:
                continue
            # try adding a random non-edge
            for _ in range(30):
                i = random.randrange(n)
                j = random.randrange(n)
                if i == j or ((adj[i] >> j) & 1):
                    continue
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                if not is_k4_free(adj):
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                    continue
                tris2 = get_triangles(adj)
                s2 = max_triangle_free(adj, tris2)
                if s2 < s:
                    s = s2
                    improved = True
                    break
                else:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
            if not improved:
                break
        return s

    # ---------- main loop ----------

    best_score = 12
    best_edges = seed_edges

    # evaluate seed
    adj = adj_from_edges(seed_edges)
    tris = get_triangles(adj)
    s = max_triangle_free(adj, tris)
    if s < best_score:
        best_score = s
        best_edges = seed_edges

    # local search on seed
    s2 = try_local_search(adj)
    if s2 < best_score:
        best_score = s2
        best_edges = edges_from_adj(adj)

    # random search
    attempts = 0
    while time.time() < deadline - 15.0:
        attempts += 1
        adj = random_k4_free()
        if not is_k4_free(adj):
            continue
        tris = get_triangles(adj)
        s = max_triangle_free(adj, tris)
        if s < best_score:
            best_score = s
            best_edges = edges_from_adj(adj)
            # local search to improve
            s2 = try_local_search(adj)
            if s2 < best_score:
                best_score = s2
                best_edges = edges_from_adj(adj)
            if best_score <= 10:
                break
        elif s <= best_score:
            # try local search even if not strictly better, in case it improves
            s2 = try_local_search(adj)
            if s2 < best_score:
                best_score = s2
                best_edges = edges_from_adj(adj)
                if best_score <= 10:
                    break

    return best_edges
