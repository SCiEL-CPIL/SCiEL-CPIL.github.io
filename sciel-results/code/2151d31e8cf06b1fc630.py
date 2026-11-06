"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
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
    def would_create_c6_fast(adj, u, v):
        """Does adding edge (u,v) create a C6? Bitset (a,d) enumeration."""
        Nu = adj[u] & ~(1 << v)
        Nv = adj[v] & ~(1 << u)

        mask_a = Nu
        while mask_a:
            bit_a = mask_a & (-mask_a)
            a = bit_a.bit_length() - 1
            mask_a ^= bit_a

            Na_base = adj[a] & ~(1 << u) & ~(1 << v)

            mask_d = Nv & ~(1 << a)  # d != a
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

                mask_b = B
                while mask_b:
                    bit_b = mask_b & (-mask_b)
                    b = bit_b.bit_length() - 1
                    mask_b ^= bit_b
                    if adj[b] & C & ~(1 << b):
                        return True
        return False

    # ---------- Full C6 existence check (for verification) ----------
    def has_c6(adj):
        """Return True if the graph contains any 6-cycle. Bitset based."""
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

    # ---------- Greedy addition with degree-pressure re-sorting ----------
    def greedy_add(adj, edge_set, edges, candidate_edges, reheap=True):
        added_any = True
        while added_any and time_ok():
            added_any = False
            remaining = []
            # Degree pressure: prefer edges touching currently low-degree vertices
            deg = [popcount(adj[i]) for i in range(n)]
            def pressure(item):
                if len(item) == 2:
                    u, v = item
                else:
                    u, v = item[1], item[2]
                return deg[u] + deg[v]
            if reheap:
                candidate_edges = sorted(candidate_edges, key=pressure)
            for item in candidate_edges:
                if len(item) == 2:
                    u, v = item
                else:
                    u, v = item[1], item[2]
                if (min(u, v), max(u, v)) in edge_set:
                    continue
                if would_create_c6_fast(adj, u, v):
                    remaining.append(item)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((min(u, v), max(u, v)))
                    edges.append([u, v])
                    added_any = True
            candidate_edges = remaining

    # ---------- Local search: edge swap to escape plateaus ----------
    def local_swap(adj, edge_set, edges, iters=400):
        """Try removing an edge and adding a different one to gain density."""
        for _ in range(iters):
            if not time_ok():
                break
            cur = len(edges)
            # Pick a random existing edge to remove
            idx = random.randrange(cur)
            ru, rv = edges[idx]
            key = (min(ru, rv), max(ru, rv))
            # Try to add a random missing edge after removal
            candidates = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (u, v) not in edge_set:
                        candidates.append((u, v))
            random.shuffle(candidates)
            improved = False
            for u, v in candidates[:80]:
                # remove ru-rv
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                edge_set.discard(key)
                edges.pop(idx)
                # try add u-v
                if not would_create_c6_fast(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((min(u, v), max(u, v)))
                    edges.append([u, v])
                    improved = True
                    break
                # revert removal
                adj[ru] |= (1 << rv)
                adj[rv] |= (1 << ru)
                edge_set.add(key)
                edges.insert(idx, [ru, rv])
            if not improved:
                # ensure ru-rv restored
                if key not in edge_set:
                    adj[ru] |= (1 << rv)
                    adj[rv] |= (1 << ru)
                    edge_set.add(key)
                    edges.append([ru, rv])

    # ---------- Spectral-guided candidate ordering ----------
    def spectral_order():
        rng = np.random.default_rng(42)
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
        return potential_edges

    # ---------- Build initial 8xK5 seed ----------
    def k5_seed():
        edges = []
        for k in range(8):
            edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
        return edges

    # ---------- Driver ----------
    best_edges = None
    best_count = -1

    def consider(edges):
        nonlocal best_edges, best_count
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]

    # Strategy A: spectral-ordered greedy
    pot = spectral_order()
    edgesA = []
    adjA = [0] * n
    esetA = set()
    greedy_add(adjA, esetA, edgesA, pot)
    local_swap(adjA, esetA, edgesA, iters=300)
    consider(edgesA)

    # Strategy B: K5 seed greedy
    edgesB = k5_seed()
    adjB = make_adj(edgesB)
    esetB = set((min(u, v), max(u, v)) for u, v in edgesB)
    allB = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in esetB]
    greedy_add(adjB, esetB, edgesB, allB)
    local_swap(adjB, esetB, edgesB, iters=300)
    consider(edgesB)

    # Strategy C: randomized restarts from best
    restarts = 260
    for restart in range(restarts):
        if not time_ok():
            break
        if restart < 6:
            current_edges = best_edges[:]
        else:
            current_edges = [e for e in best_edges if random.random() < 0.85]
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

    # Verification: ensure no C6; if found, fall back to a verified K5 seed
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
