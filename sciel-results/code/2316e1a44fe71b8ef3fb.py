import time
import random
import math

N = 18
FULL = (1 << N) - 1
DEADLINE_MARGIN = 8  # safety margin before the 150 s wall-clock kill


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    rng = random.Random()

    def is_k4_free(adj_mask):
        for a in range(N):
            for b in range(a + 1, N):
                if (adj_mask[a] >> b) & 1:
                    common = adj_mask[a] & adj_mask[b]
                    common &= ~((1 << a) | (1 << b))
                    m = common
                    c_list = []
                    while m:
                        c = (m & -m).bit_length() - 1
                        c_list.append(c)
                        m &= m - 1
                    for i in range(len(c_list)):
                        for j in range(i + 1, len(c_list)):
                            if (adj_mask[c_list[i]] >> c_list[j]) & 1:
                                return False
        return True

    def find_all_triangles(adj_mask):
        tris = []
        for a in range(N):
            for b in range(a + 1, N):
                if (adj_mask[a] >> b) & 1:
                    common = adj_mask[a] & adj_mask[b]
                    common &= ~((1 << a) | (1 << b))
                    m = common
                    while m:
                        c = (m & -m).bit_length() - 1
                        if c > b:  # ensure a < b < c
                            tris.append((a, b, c))
                        m &= m - 1
        return tris

    def max_tri_free(adj_mask):
        """Exact size of the largest triangle-free induced subgraph (branch & bound)."""
        tris = find_all_triangles(adj_mask)
        if not tris:
            return N
        best = [0]
        # Precompute, for each vertex, the set of triangles containing it (as a list of
        # the other two vertices), to speed up the "does avail contain a triangle" test.
        tris_by_v = [set() for _ in range(N)]
        for (a, b, c) in tris:
            tris_by_v[a].add((b, c))
            tris_by_v[b].add((a, c))
            tris_by_v[c].add((a, b))

        def has_triangle(avail):
            # Find any triangle fully contained in avail.
            for (a, b, c) in tris:
                if (avail >> a) & 1 and (avail >> b) & 1 and (avail >> c) & 1:
                    return (a, b, c)
            return None

        def bnb(avail):
            cnt = avail.bit_count()
            if cnt <= best[0]:
                return
            tri = has_triangle(avail)
            if tri is None:
                best[0] = cnt
                return
            a, b, c = tri
            bnb(avail & ~(1 << a))
            bnb(avail & ~(1 << b))
            bnb(avail & ~(1 << c))

        bnb(FULL)
        return best[0]

    def make_k666():
        adj_mask = [0] * N
        for i in range(N):
            for j in range(N):
                if i != j and i // 6 != j // 6:
                    adj_mask[i] |= (1 << j)
        return adj_mask

    def random_k4_free_graph():
        adj_mask = [0] * N
        for u in range(N):
            for v in range(u + 1, N):
                if rng.random() < 0.5:
                    # Check adding (u,v) does not create a K4.
                    creates_k4 = False
                    for w in range(N):
                        if w == u or w == v:
                            continue
                        if not ((adj_mask[u] >> w) & 1) or not ((adj_mask[v] >> w) & 1):
                            continue
                        for x in range(N):
                            if x in (u, v, w):
                                continue
                            if ((adj_mask[u] >> x) & 1 and (adj_mask[v] >> x) & 1
                                    and (adj_mask[w] >> x) & 1
                                    and (adj_mask[w] >> x) & 1
                                    and (adj_mask[x] >> w) & 1):
                                creates_k4 = True
                                break
                        if creates_k4:
                            break
                    if not creates_k4:
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
        return adj_mask

    def flip(adj_mask, u, v):
        if (adj_mask[u] >> v) & 1:
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
        else:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)

    def anneal(adj_mask, current_score, deadline):
        T = 1.0
        T_min = 0.01
        cooling = 0.997
        local_best = current_score
        while time.time() < deadline and T > T_min:
            u, v = rng.sample(range(N), 2)
            old_val = (adj_mask[u] >> v) & 1
            flip(adj_mask, u, v)
            if not is_k4_free(adj_mask):
                flip(adj_mask, u, v)  # revert
                T *= cooling
                continue
            new_score = max_tri_free(adj_mask)
            if new_score < current_score or (
                    T > T_min and rng.random() < math.exp((current_score - new_score) / T)):
                current_score = new_score
                if new_score < local_best:
                    local_best = new_score
            else:
                flip(adj_mask, u, v)  # revert
            T *= cooling
        return local_best

    best_score = N + 1
    best_mask = None

    # Initial structures to seed the search.
    seeds = [make_k666()]
    seeds.append(random_k4_free_graph())

    # Run several independent restarts; each gets a fair share of the budget.
    n_restarts = 6
    for r in range(n_restarts):
        if time.time() >= deadline:
            break
        if r < len(seeds):
            adj = seeds[r][:]
        else:
            adj = random_k4_free_graph()
        if not is_k4_free(adj):
            adj = make_k666()
        cur = max_tri_free(adj)
        # Per-restart deadline: keep a little margin so we always return something.
        cur_deadline = min(deadline, time.time() + (deadline - start_time) * 0.0 + 15)
        local_best = anneal(adj, cur, deadline)
        if local_best < best_score:
            best_score = local_best
            best_mask = adj[:]

    if best_mask is None:
        best_mask = make_k666()

    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if (best_mask[i] >> j) & 1:
                edges.append([i, j])
    return edges
