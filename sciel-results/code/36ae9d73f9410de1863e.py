import time
import random

def construct():
    """
    Constructs a C6-free graph on 40 vertices with a high number of edges.

    Combines:
      * structured K5-clique initialization (dense, C6-free start),
      * multiple randomized greedy restarts,
      * WalkSAT-inspired local search with violation-based edge selection,
      * a final exhaustive edge-addition pass.
    """
    N = 40
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s limit

    def check_c6_fast(u, v, adj):
        """
        Fast check: does adding edge (u,v) create a C6 that uses (u,v)?
        C6: u-a-x-y-b-v with all vertices distinct.
        """
        nu = adj[u]
        nv = adj[v]

        a_mask = nu & ~(1 << v)
        while a_mask:
            a_bit = a_mask & (-a_mask)
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit

            b_mask = nv & ~(1 << u) & ~(1 << a)
            while b_mask:
                b_bit = b_mask & (-b_mask)
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit

                x_mask = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << b)
                if not x_mask:
                    continue
                y_mask = adj[b] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                if not y_mask:
                    continue

                xm = x_mask
                while xm:
                    x_bit = xm & (-xm)
                    x = x_bit.bit_length() - 1
                    xm ^= x_bit
                    if adj[x] & (y_mask & ~(1 << x)):
                        return True
        return False

    def compute_violation_scores(adj):
        """
        For each non-edge (u,v), compute a 'violation score' that measures
        how many partial C6 paths would be completed by adding (u,v).
        Lower score = safer to add. Score 0 = safe (no C6 created).
        """
        scores = {}
        for u in range(N):
            for v in range(u + 1, N):
                if adj[u] & (1 << v):
                    continue
                # Count partial paths u-a-x-y-b-v
                count = 0
                nu = adj[u] & ~(1 << v)
                nv = adj[v] & ~(1 << u)

                a_mask = nu
                while a_mask:
                    a_bit = a_mask & (-a_mask)
                    a = a_bit.bit_length() - 1
                    a_mask ^= a_bit

                    b_mask = nv & ~(1 << a)
                    while b_mask:
                        b_bit = b_mask & (-b_bit)
                        b = b_bit.bit_length() - 1
                        b_mask ^= b_bit

                        x_mask = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << b)
                        y_mask = adj[b] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                        # Count pairs (x,y) with x in x_mask, y in y_mask, edge(x,y)
                        xm = x_mask
                        while xm:
                            x_bit = xm & (-xm)
                            x = x_bit.bit_length() - 1
                            xm ^= x_bit
                            count += (adj[x] & y_mask & ~(1 << x)).bit_count()

                scores[(u, v)] = count
        return scores

    def structured_init():
        """Initialize with 8 K5 cliques (C6-free, 80 edges)."""
        adj = [0] * N
        edge_set = set()
        for g in range(8):
            group = list(range(g * 5, g * 5 + 5))
            for i in range(5):
                for j in range(i + 1, 5):
                    u, v = group[i], group[j]
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
        return edge_set, adj

    def greedy_fill(edge_set, adj, rng):
        """Greedily add edges in random order."""
        all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        rng.shuffle(all_edges)
        for u, v in all_edges:
            if (u, v) in edge_set:
                continue
            if not check_c6_fast(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
        return edge_set, adj

    def walksat_search(edge_set, adj, rng, max_iters=5000):
        """
        WalkSAT-inspired local search:
        - Maintain violation scores for non-edges
        - At each step, either add a low-violation edge or perform a swap
        - Use random restarts within the search
        """
        current_edges = set(edge_set)
        current_adj = list(adj)
        current_count = len(current_edges)
        best_count = current_count
        best_edges = set(current_edges)
        best_adj = list(current_adj)
        no_improve = 0
        max_no_improve = 200

        # Precompute violation scores
        v_scores = compute_violation_scores(current_adj)

        for it in range(max_iters):
            if time.time() >= deadline:
                break

            improved = False

            # Try to add a safe or low-violation edge
            # First try edges with score 0 (safe)
            safe_edges = [(u, v) for (u, v), s in v_scores.items() if s == 0]
            if safe_edges:
                # Pick randomly from safe edges
                u, v = rng.choice(safe_edges)
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges.add((u, v))
                current_count += 1
                improved = True
                # Update violation scores (only need to update edges involving u or v)
                # For simplicity, recompute periodically
                if it % 50 == 0:
                    v_scores = compute_violation_scores(current_adj)
            else:
                # No safe edges; try swaps
                # Remove a random edge, see if we can add 2+ edges
                if current_count > 0:
                    e_remove = rng.choice(list(current_edges))
                    u1, v1 = e_remove

                    # Save state
                    saved_adj = list(current_adj)
                    saved_edges = set(current_edges)
                    saved_count = current_count

                    current_adj[u1] &= ~(1 << v1)
                    current_adj[v1] &= ~(1 << u1)
                    current_edges.discard(e_remove)

                    # Recompute violation scores after removal
                    v_scores_tmp = compute_violation_scores(current_adj)
                    safe_after = [(u, v) for (u, v), s in v_scores_tmp.items() 
                                  if s == 0 and (u, v) != (u1, v1) and (u, v) not in current_edges]

                    added = []
                    for _ in range(30):
                        if not safe_after:
                            break
                        u2, v2 = rng.choice(safe_after)
                        current_adj[u2] |= (1 << v2)
                        current_adj[v2] |= (1 << u2)
                        current_edges.add((u2, v2))
                        added.append((u2, v2))
                        # Remove from safe list
                        safe_after = [(u, v) for (u, v) in safe_after if (u, v) != (u2, v2)]
                        # Quick recompute
                        v_scores_tmp = compute_violation_scores(current_adj)
                        safe_after = [(u, v) for (u, v), s in v_scores_tmp.items() 
                                      if s == 0 and (u, v) not in current_edges]

                    if len(added) >= 2:
                        current_count = saved_count + len(added) - 1
                        improved = True
                        v_scores = compute_violation_scores(current_adj)
                    else:
                        # Revert
                        current_adj = saved_adj
                        current_edges = saved_edges
                        current_count = saved_count
                        v_scores = compute_violation_scores(current_adj)

            if improved:
                no_improve = 0
                if current_count > best_count:
                    best_count = current_count
                    best_edges = set(current_edges)
                    best_adj = list(current_adj)
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    break

        return best_edges, best_adj, best_count

    def final_exhaustive_add(best_edges, best_adj):
        """Add every non-edge that doesn't create a C6."""
        adj = list(best_adj)
        edges = set(best_edges)
        count = len(edges)

        changed = True
        while changed and time.time() < deadline:
            changed = False
            for u in range(N):
                for v in range(u + 1, N):
                    if (u, v) in edges:
                        continue
                    if not check_c6_fast(u, v, adj):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges.add((u, v))
                        count += 1
                        changed = True
        return edges, adj, count

    # ---- Phase 1: structured init + greedy fill (multiple seeds) ----
    best_edges = []
    best_count = 0
    best_adj = [0] * N

    for seed in range(12):
        if time.time() >= deadline:
            break
        edge_set, adj = structured_init()
        rng = random.Random(seed * 7919 + 13)
        edge_set, adj = greedy_fill(edge_set, adj, rng)
        count = len(edge_set)
        if count > best_count:
            best_count = count
            best_edges = list(edge_set)
            best_adj = list(adj)

    # ---- Phase 2: pure random greedy restarts (no structure) ----
    for seed in range(30):
        if time.time() >= deadline:
            break
        rng = random.Random(seed * 104729 + 5)
        all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        rng.shuffle(all_edges)
        adj = [0] * N
        edge_set = set()
        for u, v in all_edges:
            if not check_c6_fast(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
        count = len(edge_set)
        if count > best_count:
            best_count = count
            best_edges = list(edge_set)
            best_adj = list(adj)

    # ---- Phase 3: WalkSAT-inspired local search (multiple seeds) ----
    if best_edges and time.time() < deadline:
        adj = [0] * N
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        for ls_seed in range(20):
            if time.time() >= deadline:
                break
            rng = random.Random(100000 + ls_seed * 31)
            new_edges, new_adj, new_count = walksat_search(
                best_edges, adj, rng, max_iters=4000)
            if new_count > best_count:
                best_count = new_count
                best_edges = list(new_edges)
                best_adj = list(new_adj)
                adj = list(new_adj)

    # ---- Phase 4: final exhaustive add ----
    if best_edges and time.time() < deadline:
        new_edges, new_adj, new_count = final_exhaustive_add(best_edges, best_adj)
        if new_count > best_count:
            best_count = new_count
            best_edges = list(new_edges)
            best_adj = list(new_adj)

    # ---- Phase 5: one more WalkSAT polish if time remains ----
    if best_edges and time.time() < deadline:
        adj = [0] * N
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        rng = random.Random(987654)
        new_edges, new_adj, new_count = walksat_search(
            best_edges, adj, rng, max_iters=3000)
        if new_count > best_count:
            best_count = new_count
            best_edges = list(new_edges)
            best_adj = list(new_adj)

        # final exhaustive add again
        if time.time() < deadline:
            new_edges, new_adj, new_count = final_exhaustive_add(
                best_edges, best_adj)
            if new_count > best_count:
                best_count = new_count
                best_edges = list(new_edges)
                best_adj = list(new_adj)

    return [[u, v] for u, v in best_edges]
