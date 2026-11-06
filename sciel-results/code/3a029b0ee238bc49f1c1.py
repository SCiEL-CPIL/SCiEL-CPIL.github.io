"""Rewritten program: C4-minimizing 91-edge graph on 31 vertices."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    import numpy as np

    start = time.time()
    DEADLINE = 148.0  # 150s wall-clock limit, 2s safety
    n = 31
    target_edges = 91

    # ---- Build the polarity graph of PG(2,5): 31 vertices, C4-free ----
    # Use the standard polarity x^2 + y^2 - z^2 = 0 over GF(5)
    # Points in homogeneous coords: (x,y,z) not all zero, mod (1,1,1)
    # Two points are adjacent iff their dot product is 0 (orthogonal)
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Canonical representative: first nonzero coord = 1
                if z != 0:
                    points.append((x * pow(z, q-2, q) % q, y * pow(z, q-2, q) % q, 1))
                elif y != 0:
                    points.append((x * pow(y, q-2, q) % q, 1, 0))
                else:
                    points.append((1, 0, 0))

    # Deduplicate (should be 31 unique points)
    seen = set()
    unique_points = []
    for p in points:
        if p not in seen:
            seen.add(p)
            unique_points.append(p)
    points = unique_points
    n = len(points)  # should be 31

    poly_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = points[i], points[j]
            # Polarity: x1*x2 + y1*y2 - z1*z2 = 0 mod q
            if (p1[0]*p2[0] + p1[1]*p2[1] - p1[2]*p2[2]) % q == 0:
                poly_edges.add((i, j))

    # ---- Fast C4 counting via common-neighbor pairs ----
    # C4 count = sum over all unordered vertex pairs (u,v) of C(common(u,v), 2)
    # Each C4 has exactly 2 pairs of opposite vertices, each contributing 1.
    def count_c4(adj_sets):
        total = 0
        for u in range(n):
            nu = adj_sets[u]
            for v in range(u + 1, n):
                c = len(nu & adj_sets[v])
                total += c * (c - 1) // 2
        return total

    # ---- Incremental C4 delta for adding/removing a single edge ----
    def c4_delta_edge(u, v, adj_sets):
        """Return the number of C4s that would be created (or destroyed) by edge (u,v)."""
        cn = adj_sets[u] & adj_sets[v]
        cnt = 0
        cn_list = list(cn)
        for i in range(len(cn_list)):
            a = cn_list[i]
            for j in range(i + 1, len(cn_list)):
                b = cn_list[j]
                if b in adj_sets[a]:
                    cnt += 1
        return cnt

    # ---- Initial state: polarity graph (C4-free, ~90 edges) + best non-edge ----
    # The polarity graph of PG(2,5) with polarity x^2+y^2-z^2 is C4-free.
    # We need exactly 91 edges. If poly_edges has 90, add 1. If 93, remove 2.
    poly_edge_list = list(poly_edges)
    m = len(poly_edges)
    print(f"DEBUG: polarity graph has {m} edges")

    # Build adjacency for the polarity graph
    adj0 = [set() for _ in range(n)]
    for u, v in poly_edges:
        adj0[u].add(v)
        adj0[v].add(u)

    # Verify C4-free
    c4_poly = count_c4(adj0)
    print(f"DEBUG: C4 count in polarity graph: {c4_poly}")

    if m == 90:
        # Perfect: add the non-edge with minimum C4 delta
        all_pairs = set((i, j) for i in range(n) for j in range(i + 1, n))
        non_edges_poly = all_pairs - poly_edges
        best_add = None
        best_delta = float('inf')
        for e in non_edges_poly:
            u, v = e
            d = c4_delta_edge(u, v, adj0)
            if d < best_delta:
                best_delta = d
                best_add = e
        # Build initial state
        adj = [set(s) for s in adj0]
        u_add, v_add = best_add
        adj[u_add].add(v_add)
        adj[v_add].add(u_add)
        current_edges = set(poly_edges) | {best_add}
        current_c4 = best_delta
    elif m == 93:
        # Need to remove 2 edges to get to 91. Try all pairs.
        best_init_c4 = float('inf')
        best_init_removed = None
        for i in range(m):
            e1 = poly_edge_list[i]
            for j in range(i + 1, m):
                e2 = poly_edge_list[j]
                removed = {e1, e2}
                adj_tmp = [set() for _ in range(n)]
                for u, v in poly_edges:
                    if (u, v) not in removed and (v, u) not in removed:
                        adj_tmp[u].add(v)
                        adj_tmp[v].add(u)
                c4 = count_c4(adj_tmp)
                if c4 < best_init_c4:
                    best_init_c4 = c4
                    best_init_removed = (e1, e2)
        e1, e2 = best_init_removed
        adj = [set() for _ in range(n)]
        for u, v in poly_edges:
            if (u, v) != e1 and (u, v) != e2:
                adj[u].add(v)
                adj[v].add(u)
        current_c4 = best_init_c4
        current_edges = set(poly_edges) - {e1, e2}
    else:
        # Fallback: take first 91 edges
        current_edges = set(poly_edge_list[:91])
        adj = [set() for _ in range(n)]
        for u, v in current_edges:
            adj[u].add(v)
            adj[v].add(u)
        current_c4 = count_c4(adj)

    all_pairs = set((i, j) for i in range(n) for j in range(i + 1, n))
    non_edges = all_pairs - current_edges
    print(f"DEBUG: initial C4 = {current_c4}, edges = {len(current_edges)}")

    # ---- Phase 1: Greedy local search (deterministic) ----
    # From the near-optimal starting point, do exhaustive 1-edge swaps.
    best_c4 = current_c4
    best_edges = set(current_edges)
    all_pairs_set = all_pairs

    def full_greedy_pass(adj, current_edges, non_edges, current_c4_val, deadline):
        """Do one full pass of best-improvement 1-edge swaps."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_swap = None
            best_new_c4 = current_c4_val
            edges_list = list(current_edges)
            non_edges_list = list(non_edges)

            for e_remove in edges_list:
                if time.time() > deadline:
                    break
                u1, v1 = e_remove
                d_rem = c4_delta_edge(u1, v1, adj)
                # Temporarily remove
                adj[u1].discard(v1)
                adj[v1].discard(u1)
                for e_add in non_edges_list:
                    u2, v2 = e_add
                    d_add = c4_delta_edge(u2, v2, adj)
                    new_c4 = current_c4_val - d_rem + d_add
                    if new_c4 < best_new_c4:
                        best_new_c4 = new_c4
                        best_swap = (e_remove, e_add)
                # Restore
                adj[u1].add(v1)
                adj[v1].add(u1)

            if best_swap:
                e_remove, e_add = best_swap
                u1, v1 = e_remove
                u2, v2 = e_add
                adj[u1].discard(v1)
                adj[v1].discard(u1)
                adj[u2].add(v2)
                adj[v2].add(u2)
                current_edges.discard(e_remove)
                current_edges.add(e_add)
                non_edges.discard(e_add)
                non_edges.add(e_remove)
                current_c4_val = best_new_c4
                improved = True
        return current_c4_val

    deadline1 = DEADLINE - 20
    current_c4 = full_greedy_pass(adj, current_edges, non_edges, current_c4, deadline1)
    if current_c4 < best_c4:
        best_c4 = current_c4
        best_edges = set(current_edges)

    # ---- Phase 2: Simulated annealing for escape from local optima ----
    if time.time() < DEADLINE - 10:
        adj = [set() for _ in range(n)]
        current_edges = set(best_edges)
        for u, v in current_edges:
            adj[u].add(v)
            adj[v].add(u)
        non_edges = all_pairs_set - current_edges
        current_c4 = best_c4

        rng = random.Random(42)
        T = 2.0
        T_min = 0.05
        cooling = 0.9998
        step = 0
        max_steps = 200000

        while step < max_steps and time.time() < DEADLINE - 5:
            e_remove = rng.choice(list(current_edges))
            e_add = rng.choice(list(non_edges))

            u1, v1 = e_remove
            u2, v2 = e_add

            # Compute swap delta
            d_rem = c4_delta_edge(u1, v1, adj)
            adj[u1].discard(v1)
            adj[v1].discard(u1)
            d_add = c4_delta_edge(u2, v2, adj)
            adj[u1].add(v1)
            adj[v1].add(u1)

            new_c4 = current_c4 - d_rem + d_add

            if new_c4 <= current_c4:
                # Accept
                adj[u1].discard(v1)
                adj[v1].discard(u1)
                adj[u2].add(v2)
                adj[v2].add(u2)
                current_edges.discard(e_remove)
                current_edges.add(e_add)
                non_edges.discard(e_add)
                non_edges.add(e_remove)
                current_c4 = new_c4
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_edges = set(current_edges)
            elif T > T_min:
                delta = new_c4 - current_c4
                if delta / T < 50:
                    if rng.random() < np.exp(-delta / T):
                        adj[u1].discard(v1)
                        adj[v1].discard(u1)
                        adj[u2].add(v2)
                        adj[v2].add(u2)
                        current_edges.discard(e_remove)
                        current_edges.add(e_add)
                        non_edges.discard(e_add)
                        non_edges.add(e_remove)
                        current_c4 = new_c4

            T *= cooling
            step += 1

        # Final greedy pass after SA
        if time.time() < DEADLINE - 3:
            adj = [set() for _ in range(n)]
            current_edges = set(best_edges)
            for u, v in current_edges:
                adj[u].add(v)
                adj[v].add(u)
            non_edges = all_pairs_set - current_edges
            current_c4 = best_c4
            current_c4 = full_greedy_pass(adj, current_edges, non_edges, current_c4, DEADLINE - 1)
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edges = set(current_edges)

    # ---- Return best result ----
    result_edges = best_edges
    # Safety: ensure exactly 91 edges
    if len(result_edges) != target_edges:
        if len(result_edges) > target_edges:
            result_edges = set(list(result_edges)[:target_edges])
        else:
            for p in sorted(all_pairs_set):
                if len(result_edges) >= target_edges:
                    break
                if p not in result_edges:
                    result_edges.add(p)

    return [[u, v] for u, v in sorted(result_edges)]
# EVOLVE-BLOCK-END
