"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n = 40

    def time_ok():
        return time.time() < deadline

    # ---------- Bitset adjacency helpers ----------
    def make_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def popcount(x):
        return x.bit_count()

    # ---------- Fast incremental C6 check ----------
    def would_create_c6(adj, u, v):
        """Does adding edge (u,v) create a C6? Bitset enumeration."""
        Nu = adj[u] & ~(1 << v)
        Nv = adj[v] & ~(1 << u)

        # Choose the smaller neighbor set to iterate over
        if popcount(Nu) > popcount(Nv):
            Nu, Nv = Nv, Nu

        mask_a = Nu
        while mask_a:
            bit_a = mask_a & (-mask_a)
            a = bit_a.bit_length() - 1
            mask_a ^= bit_a

            Na_base = adj[a] & ~(1 << u) & ~(1 << v)
            mask_d = Nv & ~(1 << a)
            while mask_d:
                bit_d = mask_d & (-mask_d)
                d = bit_d.bit_length() - 1
                mask_d ^= bit_d

                B = Na_base & ~(1 << d)
                if B == 0:
                    continue
                C = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                if C == 0:
                    continue
                # Early termination: if B and C are disjoint, no b in B can connect to C
                if B & C == 0:
                    continue

                mask_b = B
                while mask_b:
                    bit_b = mask_b & (-mask_b)
                    b = bit_b.bit_length() - 1
                    mask_b ^= bit_b
                    if adj[b] & C & ~(1 << b):
                        return True
        return False

    # ---------- Full C6 existence check ----------
    def has_c6(adj):
        for u in range(n):
            N_u = adj[u]
            if N_u == 0:
                continue
            neighbors_u = []
            mask = N_u
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                neighbors_u.append(v)
                mask ^= bit
            for i in range(len(neighbors_u)):
                v = neighbors_u[i]
                for j in range(i + 1, len(neighbors_u)):
                    z = neighbors_u[j]
                    Nv_excl_u = adj[v] & ~(1 << u)
                    Nz_excl_u = adj[z] & ~(1 << u)
                    mask_w = Nv_excl_u & ~(1 << z)
                    while mask_w:
                        bit_w = mask_w & (-mask_w)
                        w = bit_w.bit_length() - 1
                        mask_w ^= bit_w
                        if w == u or w == v or w == z:
                            continue
                        Nw_excl = adj[w] & ~(1 << u) & ~(1 << v)
                        mask_y = Nz_excl_u & ~(1 << w)
                        while mask_y:
                            bit_y = mask_y & (-mask_y)
                            y = bit_y.bit_length() - 1
                            mask_y ^= bit_y
                            if y == u or y == v or y == z or y == w:
                                continue
                            Ny_excl = adj[y] & ~(1 << u) & ~(1 << z)
                            common = Nw_excl & Ny_excl
                            common &= ~(1 << w) & ~(1 << y) & ~(1 << z)
                            if common:
                                return True
        return False

    # ---------- Greedy addition with degree-based ordering ----------
    def greedy_add(adj, edge_set, edges, candidate_edges, reheap=True):
        added_any = True
        while added_any and time_ok():
            added_any = False
            remaining = []
            deg = [popcount(adj[i]) for i in range(n)]
            def pressure(item):
                u, v = item
                return deg[u] + deg[v]
            if reheap:
                candidate_edges = sorted(candidate_edges, key=pressure, reverse=True)
            for item in candidate_edges:
                u, v = item
                if (min(u, v), max(u, v)) in edge_set:
                    continue
                if would_create_c6(adj, u, v):
                    remaining.append(item)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((min(u, v), max(u, v)))
                    edges.append([u, v])
                    added_any = True
            candidate_edges = remaining

    # ---------- Local search: edge swap ----------
    def local_swap(adj, edge_set, edges, iters=400):
        for _ in range(iters):
            if not time_ok():
                break
            cur = len(edges)
            if cur == 0:
                break
            idx = random.randrange(cur)
            ru, rv = edges[idx]
            key = (min(ru, rv), max(ru, rv))
            # Build non-edge list once per call, sample per iteration
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n)
                         if (u, v) not in edge_set]
            if not non_edges:
                break
            # Sample up to 60 candidates randomly
            sample_size = min(60, len(non_edges))
            candidates = random.sample(non_edges, sample_size)
            improved = False
            for u, v in candidates:
                if (u, v) in edge_set:
                    continue
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                edge_set.discard(key)
                edges.pop(idx)
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((min(u, v), max(u, v)))
                    edges.append([u, v])
                    improved = True
                    break
                adj[ru] |= (1 << rv)
                adj[rv] |= (1 << ru)
                edge_set.add(key)
                edges.insert(idx, [ru, rv])
            if not improved:
                if key not in edge_set:
                    adj[ru] |= (1 << rv)
                    adj[rv] |= (1 << ru)
                    edge_set.add(key)
                    edges.append([ru, rv])

    # ---------- Spectral-guided candidate ordering ----------
    def spectral_order(seed=42):
        rng = np.random.default_rng(seed)
        adj_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            candidates = list(range(n))
            candidates.remove(i)
            rng.shuffle(candidates)
            for j in candidates[:4]:
                if adj_matrix[i][j] == 0:
                    adj_matrix[i][j] = 1
                    adj_matrix[j][i] = 1
            deg = adj_matrix[i].sum()
            if deg < 4:
                for j in range(n):
                    if i != j and adj_matrix[i][j] == 0:
                        adj_matrix[i][j] = 1
                        adj_matrix[j][i] = 1
                        deg += 1
                        if deg >= 4:
                            break
        eigenvalues, eigenvectors = np.linalg.eigh(adj_matrix)
        target = np.sqrt(5)
        idx = np.argmin(np.abs(eigenvalues - target))
        spec_vec = eigenvectors[:, idx]
        potential_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                w = abs(spec_vec[i] * spec_vec[j])
                potential_edges.append((w, i, j))
        potential_edges.sort(reverse=True)
        return [(u, v) for w, u, v in potential_edges]

    # ---------- Build initial 8xK5 seed ----------
    def k5_seed():
        edges = []
        for k in range(8):
            edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
        return edges

    # ---------- Degree-based initialization ----------
    def degree_init():
        """Build a random graph with degree ~5, then use greedy to complete."""
        edges = []
        edge_set = set()
        adj = [0] * n
        # Randomly add edges to get average degree ~5
        all_possible = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_possible)
        for u, v in all_possible:
            if popcount(adj[u]) < 5 and popcount(adj[v]) < 5:
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
                    edges.append([u, v])
        return edges, edge_set, adj

    # ---------- Driver ----------
    best_edges = None
    best_count = -1

    def consider(edges):
        nonlocal best_edges, best_count
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]

    # Strategy A: spectral-ordered greedy
    if time_ok():
        pot = spectral_order()
        edgesA = []
        adjA = [0] * n
        esetA = set()
        greedy_add(adjA, esetA, edgesA, pot)
        local_swap(adjA, esetA, edgesA, iters=500)
        consider(edgesA)

    # Strategy B: K5 seed greedy
    if time_ok():
        edgesB = k5_seed()
        adjB = make_adj(edgesB)
        esetB = set((min(u, v), max(u, v)) for u, v in edgesB)
        allB = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in esetB]
        greedy_add(adjB, esetB, edgesB, allB)
        local_swap(adjB, esetB, edgesB, iters=500)
        consider(edgesB)

    # Strategy C: degree-based initialization
    if time_ok():
        edgesC, esetC, adjC = degree_init()
        remainingC = [(u, v) for u in range(n) for v in range(u + 1, n)
                      if (u, v) not in esetC]
        random.shuffle(remainingC)
        greedy_add(adjC, esetC, edgesC, remainingC, reheap=False)
        local_swap(adjC, esetC, edgesC, iters=300)
        consider(edgesC)

    # Strategy D: randomized restarts from best
    restarts = 300
    for restart in range(restarts):
        if not time_ok():
            break
        if restart < 5:
            current_edges = best_edges[:]
        else:
            current_edges = [e for e in best_edges if random.random() < 0.88]
        current_edge_set = set((min(u, v), max(u, v)) for u, v in current_edges)
        current_adj = make_adj(current_edges)
        remaining = [(u, v) for u in range(n) for v in range(u + 1, n)
                     if (u, v) not in current_edge_set]
        random.shuffle(remaining)
        greedy_add(current_adj, current_edge_set, current_edges, remaining, reheap=False)
        local_swap(current_adj, current_edge_set, current_edges, iters=150)
        consider(current_edges)

    # Final polish
    if best_edges is not None:
        final_edge_set = set((min(u, v), max(u, v)) for u, v in best_edges)
        final_adj = make_adj(best_edges)
        final_remaining = [(u, v) for u in range(n) for v in range(u + 1, n)
                           if (u, v) not in final_edge_set]
        greedy_add(final_adj, final_edge_set, best_edges, final_remaining)

    # Verification
    if best_edges is not None:
        vadj = make_adj(best_edges)
        if has_c6(vadj):
            fb = k5_seed()
            fadj = make_adj(fb)
            fset = set((min(u, v), max(u, v)) for u, v in fb)
            fres = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in fset]
            greedy_add(fadj, fset, fb, fres)
            vadj = make_adj(fb)
            if has_c6(vadj):
                return k5_seed()
            return fb
        return best_edges
    return k5_seed()
# EVOLVE-BLOCK-END
