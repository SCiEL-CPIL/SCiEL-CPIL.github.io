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
        
        # Build adjacency bitmask for triangle membership
        tri_masks = [(1 << a) | (1 << b) | (1 << c) for (a, b, c) in tris]
        
        def has_triangle(avail):
            for tm in tri_masks:
                if tm & avail == tm:
                    return tm
            return None
        
        def bnb(avail, depth=0):
            cnt = avail.bit_count()
            if cnt <= best[0]:
                return
            tri = has_triangle(avail)
            if tri is None:
                best[0] = cnt
                return
            # Branch on the three vertices of the triangle
            # Try removing each vertex
            a_bit = tri & -tri
            a = a_bit.bit_length() - 1
            tri_rest = tri ^ a_bit
            b_bit = tri_rest & -tri_rest
            b = b_bit.bit_length() - 1
            c = (tri ^ a_bit ^ b_bit).bit_length() - 1
            
            # Heuristic: try removing the vertex with highest degree in avail first
            deg_a = (avail & adj_mask[a]).bit_count()
            deg_b = (avail & adj_mask[b]).bit_count()
            deg_c = (avail & adj_mask[c]).bit_count()
            
            verts = [(deg_a, a), (deg_b, b), (deg_c, c)]
            verts.sort(reverse=True)
            
            for _, v in verts:
                bnb(avail & ~(1 << v), depth + 1)
                if cnt - 1 <= best[0]:
                    return

        bnb(FULL)
        return best[0]

    def make_k666():
        adj_mask = [0] * N
        for i in range(N):
            for j in range(N):
                if i != j and i // 6 != j // 6:
                    adj_mask[i] |= (1 << j)
        return adj_mask

    def make_turan_like():
        """Turan-like graph: complete multipartite with balanced parts."""
        # For K4-free, use Turan graph T(3,18) = complete 3-partite with parts of size 6
        # This is K4-free (no K4 in complete 3-partite)
        adj_mask = [0] * N
        for i in range(N):
            for j in range(i + 1, N):
                if i // 6 != j // 6:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        return adj_mask

    def make_cyclic():
        """Cyclic graph with some structure to break triangles."""
        adj_mask = [0] * N
        # Start with a cycle
        for i in range(N):
            j = (i + 1) % N
            adj_mask[i] |= (1 << j)
            adj_mask[j] |= (1 << i)
        # Add some chords carefully to avoid K4
        for i in range(N):
            for offset in [2, 3]:
                j = (i + offset) % N
                if i != j:
                    # Check if adding this edge creates a K4
                    creates_k4 = False
                    for w in range(N):
                        if w == i or w == j:
                            continue
                        if not ((adj_mask[i] >> w) & 1) or not ((adj_mask[j] >> w) & 1):
                            continue
                        for x in range(N):
                            if x in (i, j, w):
                                continue
                            if ((adj_mask[i] >> x) & 1 and (adj_mask[j] >> x) & 1
                                    and (adj_mask[w] >> x) & 1
                                    and (adj_mask[x] >> w) & 1):
                                creates_k4 = True
                                break
                        if creates_k4:
                            break
                    if not creates_k4:
                        adj_mask[i] |= (1 << j)
                        adj_mask[j] |= (1 << i)
        return adj_mask

    def random_k4_free_graph(density=0.5):
        adj_mask = [0] * N
        for u in range(N):
            for v in range(u + 1, N):
                if rng.random() < density:
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

    def local_search_polish(adj_mask, deadline):
        """Final polish: try all single-edge flips and keep improvements."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            cur_score = max_tri_free(adj_mask)
            for u in range(N):
                for v in range(u + 1, N):
                    if time.time() >= deadline:
                        return adj_mask, cur_score
                    old_val = (adj_mask[u] >> v) & 1
                    flip(adj_mask, u, v)
                    if is_k4_free(adj_mask):
                        new_score = max_tri_free(adj_mask)
                        if new_score < cur_score:
                            cur_score = new_score
                            improved = True
                        else:
                            flip(adj_mask, u, v)  # revert
                    else:
                        flip(adj_mask, u, v)  # revert
        return adj_mask, cur_score

    def anneal(adj_mask, current_score, deadline):
        T = 1.0
        T_min = 0.005
        cooling = 0.995
        local_best = current_score
        no_improve_count = 0
        while time.time() < deadline and T > T_min:
            u, v = rng.sample(range(N), 2)
            old_val = (adj_mask[u] >> v) & 1
            flip(adj_mask, u, v)
            if not is_k4_free(adj_mask):
                flip(adj_mask, u, v)  # revert
                T *= cooling
                no_improve_count += 1
                continue
            new_score = max_tri_free(adj_mask)
            if new_score < current_score or (
                    T > T_min and rng.random() < math.exp((current_score - new_score) / T)):
                current_score = new_score
                no_improve_count = 0
                if new_score < local_best:
                    local_best = new_score
            else:
                flip(adj_mask, u, v)  # revert
                no_improve_count += 1
            T *= cooling
        return local_best

    best_score = N + 1
    best_mask = None

    # Initial structures to seed the search - diverse starting points
    seeds = [make_k666(), make_turan_like(), make_cyclic()]
    for d in [0.3, 0.5, 0.7]:
        seeds.append(random_k4_free_graph(density=d))

    # Run several independent restarts; each gets a fair share of the budget.
    n_restarts = 8
    for r in range(n_restarts):
        if time.time() >= deadline:
            break
        if r < len(seeds):
            adj = seeds[r][:]
        else:
            adj = random_k4_free_graph(density=0.5)
        if not is_k4_free(adj):
            adj = make_k666()
        cur = max_tri_free(adj)
        local_best = anneal(adj, cur, deadline)
        if local_best < best_score:
            best_score = local_best
            best_mask = adj[:]

    # Final polish pass on the best solution found
    if best_mask is not None and time.time() < deadline - 2:
        polished, polished_score = local_search_polish(best_mask[:], deadline - 1)
        if polished_score < best_score:
            best_score = polished_score
            best_mask = polished

    if best_mask is None:
        best_mask = make_k666()

    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if (best_mask[i] >> j) & 1:
                edges.append([i, j])
    return edges
