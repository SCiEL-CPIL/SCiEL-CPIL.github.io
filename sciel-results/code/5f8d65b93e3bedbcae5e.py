import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118  # safety margin under 120s limit

    n = 40
    all_mask = (1 << n) - 1

    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_pairs = len(all_pairs)

    # ---------- Fast C6 detection and edge-addability ----------
    # To check if adding edge (u, v) creates a C6, we check if there exists a
    # path of length 5 between u and v (using current adjacency without the
    # new edge). We use bit-mask intersections for speed.

    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) creates a C6. Returns True if safe."""
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)
        if nu == 0 or nv == 0:
            return True

        # Precompute: for each vertex w, N(w) without u,v
        # We need path u-a-b-c-d-v, all distinct
        # a in N(u), d in N(v), b in N(a)\{u,v,d}, c in N(d)\{u,v,a}, b~c

        # Strategy: iterate over pairs (a,d) where a in N(u), d in N(v), a!=d
        # Then check if there exist b,c with b~N(a)\{u,v,a,d}, c~N(d)\{u,v,a,d}, b~c

        # For speed, precompute N_excl[w] = adj[w] & ~(1<<u) & ~(1<<v) for all w
        # But that's 40 ops per call. Instead, compute on the fly.

        # Iterate a in N(u)
        am = nu
        while am:
            lb_a = am & (-am)
            ai = lb_a.bit_length() - 1
            am ^= lb_a

            # d in N(v) \ {a}
            dm = nv & ~lb_a
            while dm:
                lb_d = dm & (-dm)
                di = lb_d.bit_length() - 1
                dm ^= lb_d

                # b candidates: N(a) \ {u,v,a,d}
                b_cand = adj[ai] & ~(1 << u) & ~(1 << v) & ~lb_a & ~lb_d
                if b_cand == 0:
                    continue

                # c candidates: N(d) \ {u,v,a,d}
                c_cand = adj[di] & ~(1 << u) & ~(1 << v) & ~lb_a & ~lb_d
                if c_cand == 0:
                    continue

                # Check: exists b in b_cand, c in c_cand with b~c
                # Union of N(b) for b in b_cand, intersect c_cand
                bm = b_cand
                while bm:
                    lb_b = bm & (-bm)
                    bi = lb_b.bit_length() - 1
                    bm ^= lb_b
                    if adj[bi] & c_cand:
                        return False
        return True

    def count_edges(adj):
        c = 0
        for i in range(n):
            c += adj[i].bit_count()
        return c // 2

    def edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def find_one_c6(adj):
        """Return one C6 as list of 6 edges [(u0,u1),...], or None."""
        for u in range(n):
            nu = adj[u]
            # v must be non-adjacent to u
            nv_cand = all_mask & ~(1 << u) & ~nu
            v = nv_cand
            while v:
                lb_v = v & (-v)
                vi = lb_v.bit_length() - 1
                v ^= lb_v

                # Path u-a-b-c-d-v
                am = nu
                while am:
                    lb_a = am & (-am)
                    ai = lb_a.bit_length() - 1
                    am ^= lb_a

                    dm = adj[vi] & ~lb_a
                    while dm:
                        lb_d = dm & (-dm)
                        di = lb_d.bit_length() - 1
                        dm ^= lb_d

                        b_cand = adj[ai] & ~(1 << u) & ~lb_v & ~lb_a & ~lb_d
                        c_cand = adj[di] & ~(1 << u) & ~lb_v & ~lb_a & ~lb_d

                        bm = b_cand
                        found = None
                        while bm:
                            lb_b = bm & (-bm)
                            bi = lb_b.bit_length() - 1
                            bm ^= lb_b
                            cm = adj[bi] & c_cand
                            if cm:
                                lb_c = cm & (-cm)
                                ci = lb_c.bit_length() - 1
                                found = (bi, ci)
                                break
                        if found:
                            bi, ci = found
                            return [(u, ai), (ai, bi), (bi, ci), (ci, di), (di, vi), (vi, u)]
        return None

    # ---------- Parameterized model ----------
    # Model: 8 blocks of 5 vertices. Parameters:
    #   intra_p[k]: probability of edges within block k
    #   inter_p[k1][k2]: probability of edges between block k1 and k2
    # We'll sample from this model, prune, then local-search.
    # To make this more "model-and-optimize", we can also do a greedy
    # construction guided by a score function.

    def build_from_params(rng, intra_p, inter_p):
        adj = [0] * n
        # Intra-block
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    if rng.random() < intra_p[k]:
                        u = base + a
                        v = base + b
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        # Inter-block
        for k1 in range(8):
            for k2 in range(k1 + 1, 8):
                p = inter_p[k1][k2]
                base1 = 5 * k1
                base2 = 5 * k2
                for a in range(5):
                    for b in range(5):
                        if rng.random() < p:
                            u = base1 + a
                            v = base2 + b
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
        return adj

    def prune_c6s(adj, rng, max_iter=200):
        """Remove edges to break C6s. Returns (adj, num_removed)."""
        removed = 0
        for _ in range(max_iter):
            if time.time() > deadline - 25:
                break
            c6 = find_one_c6(adj)
            if c6 is None:
                break
            # Remove a random edge from the C6
            idx = rng.randint(0, 5)
            u, v = c6[idx]
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            removed += 1
        return adj, removed

    def greedy_add(adj, rng):
        """Try to add edges in random order, skipping those that create C6."""
        count = count_edges(adj)
        # Build list of non-edges
        non_edges = []
        for u, v in all_pairs:
            if not ((adj[u] >> v) & 1):
                non_edges.append((u, v))
        rng.shuffle(non_edges)
        for u, v in non_edges:
            if time.time() > deadline - 8:
                break
            if not ((adj[u] >> v) & 1):
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    count += 1
        return adj, count

    def local_search(adj, rng, budget_time):
        """Local search with add, swap, and 2-opt moves."""
        count = count_edges(adj)
        best_adj = adj[:]
        best_count = count

        # Maintain edge lists
        existing = []
        non_existing = []
        for u, v in all_pairs:
            if (adj[u] >> v) & 1:
                existing.append((u, v))
            else:
                non_existing.append((u, v))

        # Precompute adjacency as list of sets for faster lookup in some ops
        # Actually bitmasks are fine.

        no_improve = 0
        max_no_improve = 300

        while time.time() < budget_time:
            if not non_existing:
                break

            r = rng.random()
            if r < 0.45:
                # Add move: try to add a random non-edge
                idx = rng.randint(0, len(non_existing) - 1)
                u, v = non_existing[idx]
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    count += 1
                    non_existing.pop(idx)
                    existing.append((u, v))
                    no_improve = 0
                    if count > best_count:
                        best_count = count
                        best_adj = adj[:]
                else:
                    no_improve += 1
            elif r < 0.75:
                # Swap: remove one edge, add another
                if not existing or not non_existing:
                    no_improve += 1
                    continue
                idx_rem = rng.randint(0, len(existing) - 1)
                u1, v1 = existing[idx_rem]
                idx_add = rng.randint(0, len(non_existing) - 1)
                u2, v2 = non_existing[idx_add]

                # Remove u1-v1
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)

                if can_add_edge(adj, u2, v2):
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)

                    existing.pop(idx_rem)
                    non_existing.append((u1, v1))
                    non_existing.pop(idx_add)
                    existing.append((u2, v2))
                    no_improve = 0
                    if count > best_count:
                        best_count = count
                        best_adj = adj[:]
                else:
                    # Reject
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    no_improve += 1
            else:
                # 2-opt: remove two edges, add two edges (net zero or +1)
                if len(existing) < 2 or len(non_existing) < 2:
                    no_improve += 1
                    continue
                idx_rem1 = rng.randint(0, len(existing) - 1)
                u1, v1 = existing[idx_rem1]
                idx_rem2 = rng.randint(0, len(existing) - 1)
                if idx_rem2 == idx_rem1:
                    no_improve += 1
                    continue
                u2, v2 = existing[idx_rem2]

                idx_add1 = rng.randint(0, len(non_existing) - 1)
                wa, xa = non_existing[idx_add1]
                idx_add2 = rng.randint(0, len(non_existing) - 1)
                if idx_add2 == idx_add1:
                    no_improve += 1
                    continue
                wb, xb = non_existing[idx_add2]

                # Remove both edges
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                adj[u2] &= ~(1 << v2)
                adj[v2] &= ~(1 << u2)

                # Try to add both new edges
                ok1 = can_add_edge(adj, wa, xa)
                if ok1:
                    adj[wa] |= (1 << xa)
                    adj[xa] |= (1 << wa)
                    ok2 = can_add_edge(adj, wb, xb)
                    if ok2:
                        adj[wb] |= (1 << xb)
                        adj[xb] |= (1 << wb)

                        # Update lists
                        # Remove the two removed edges from existing
                        # Find indices carefully
                        if idx_rem1 < idx_rem2:
                            existing.pop(idx_rem2)
                            existing.pop(idx_rem1)
                        else:
                            existing.pop(idx_rem1)
                            existing.pop(idx_rem2)

                        # Add the two new edges to existing
                        existing.append((wa, xa))
                        existing.append((wb, xb))

                        # Update non_existing: remove the two added, add the two removed
                        if idx_add1 < idx_add2:
                            non_existing.pop(idx_add2)
                            non_existing.pop(idx_add1)
                        else:
                            non_existing.pop(idx_add1)
                            non_existing.pop(idx_add2)

                        non_existing.append((u1, v1))
                        non_existing.append((u2, v2))

                        no_improve = 0
                        if count > best_count:
                            best_count = count
                            best_adj = adj[:]
                    else:
                        # Undo second add
                        adj[wa] &= ~(1 << xa)
                        adj[xa] &= ~(1 << wa)
                        # Restore removed edges
                        adj[u1] |= (1 << v1)
                        adj[v1] |= (1 << u1)
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        no_improve += 1
                else:
                    # Restore
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    no_improve += 1

            if no_improve > max_no_improve:
                # Try a few consecutive greedy adds to escape local optimum
                no_improve = 0
                for _ in range(10):
                    if not non_existing:
                        break
                    idx = rng.randint(0, len(non_existing) - 1)
                    u, v = non_existing[idx]
                    if can_add_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        count += 1
                        non_existing.pop(idx)
                        existing.append((u, v))
                        if count > best_count:
                            best_count = count
                            best_adj = adj[:]
                    else:
                        break

        return best_adj, best_count

    # ---------- Main search ----------
    best_adj = None
    best_count = 0

    # Try multiple parameter settings
    param_families = []

    # Family 1: high intra, moderate inter
    intra_high = [0.7, 0.7, 0.7, 0.7, 0.6, 0.6, 0.6, 0.6]
    inter_mod = [[0.0]*8 for _ in range(8)]
    for i in range(8):
        for j in range(8):
            if i != j:
                inter_mod[i][j] = 0.15
    param_families.append((intra_high, inter_mod))

    # Family 2: moderate intra, lower inter
    intra_med = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    inter_low = [[0.0]*8 for _ in range(8)]
    for i in range(8):
        for j in range(8):
            if i != j:
                inter_low[i][j] = 0.10
    param_families.append((intra_med, inter_low))

    # Family 3: random parameters
    for _ in range(4):
        intra_r = [random.random() * 0.6 + 0.2 for _ in range(8)]
        inter_r = [[0.0]*8 for _ in range(8)]
        for i in range(8):
            for j in range(8):
                if i != j:
                    inter_r[i][j] = random.random() * 0.15 + 0.05
        param_families.append((intra_r, inter_r))

    random.seed(12345)
    rng = random.Random(54321)

    # Phase 1: Generate diverse starting points
    for fam_idx, (intra_p, inter_p) in enumerate(param_families):
        if time.time() > deadline - 50:
            break
        for restart in range(3):
            if time.time() > deadline - 45:
                break
            rng2 = random.Random(fam_idx * 100 + restart)
            adj = build_from_params(rng2, intra_p, inter_p)
            adj, _ = prune_c6s(adj, rng2)
            adj, cnt = greedy_add(adj, rng2)
            if cnt > best_count:
                best_count = cnt
                best_adj = adj[:]

    # If no good solution yet, fallback to K5 base
    if best_adj is None or best_count < 80:
        best_adj = [0] * n
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    u = base + a
                    v = base + b
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
        best_count = count_edges(best_adj)

    # Phase 2: Local search with time budget
    remaining_time = deadline - time.time() - 5
    if remaining_time > 10:
        # Split time: 70% for main local search, 30% for final polish
        ls_time = time.time() + remaining_time * 0.7
        best_adj, best_count = local_search(best_adj, random.Random(999), ls_time)

    # Phase 3: Final greedy add pass
    if time.time() < deadline - 3:
        best_adj, best_count = greedy_add(best_adj, random.Random(777))

    # Phase 4: Final short local search
    if time.time() < deadline - 2:
        best_adj, best_count = local_search(best_adj, random.Random(888), deadline - 2)

    if not best_adj:
        best_adj = [0] * n
        best_count = 0

    result = edges_from_adj(best_adj)
    return result
