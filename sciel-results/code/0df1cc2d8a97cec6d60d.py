import time
import random

def construct():
    """
    Constructs a C6-free graph on 40 vertices with a high number of edges.

    Combines:
      * structured K5-clique initialization (dense, C6-free start),
      * multiple randomized greedy restarts,
      * local search with add / 1-swap / 2-add-1-remove moves,
      * a final exhaustive edge-addition pass accelerated by a
        precomputed "forbidden pair" set (pairs that would create a C6
        if added to the current graph).
    """
    N = 40
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s limit

    def check_c6_fast(u, v, adj):
        """
        Fast check: does adding edge (u,v) create a C6 that uses (u,v)?
        C6: u-a-x-y-b-v with all vertices distinct.
        For a in N(u)\\{v}, b in N(v)\\{u,a}:
          need x in N(a)\\{u,v,b}, y in N(b)\\{u,v,a} with (x,y) edge, x!=y.
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

    def would_create_c6(u, v, adj):
        """
        Full check: does edge (u,v) belong to ANY C6 in the current graph?
        (u,v) may or may not already be present; we test the 6-cycle condition.
        C6: u-a-b-c-d-v with a,b,c,d distinct and not in {u,v}.
        """
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)

        a_mask = nu
        while a_mask:
            a_bit = a_mask & (-a_mask)
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit

            # b in N(a) \ {u,v}
            b_mask = adj[a] & ~(1 << u) & ~(1 << v)
            while b_mask:
                b_bit = b_mask & (-b_mask)
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit
                if b == a:
                    continue

                # c in N(b) \ {u,v,a}
                c_mask = adj[b] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                while c_mask:
                    c_bit = c_mask & (-c_mask)
                    c = c_bit.bit_length() - 1
                    c_mask ^= c_bit
                    if c == b:
                        continue

                    # d in N(c) \ {u,v,b}  and  d in N(v)
                    d_mask = adj[c] & adj[v] & ~(1 << u) & ~(1 << v) & ~(1 << b)
                    if d_mask:
                        return True
        return False

    def compute_forbidden(adj):
        """
        Return set of (u,v) pairs (u<v) that would create a C6 if added
        to the graph described by adj. Used to accelerate the final pass.
        """
        forbidden = set()
        for u in range(N):
            for v in range(u + 1, N):
                if would_create_c6(u, v, adj):
                    forbidden.add((u, v))
        return forbidden

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

    def local_search(edge_set, adj, rng, max_iters=3000):
        """
        Local search with three move types:
          1. add a random non-edge,
          2. 1-swap (remove e1, add e2),
          3. 2-add-1-remove (remove e1, add e2 and e3) -> net +1.
        """
        current_edges = set(edge_set)
        current_adj = list(adj)
        current_count = len(current_edges)
        no_improve = 0
        max_no_improve = 120

        for it in range(max_iters):
            if time.time() >= deadline:
                break

            improved = False

            # --- Move 1: add a random non-edge ---
            for _ in range(12):
                u, v = rng.randint(0, N - 1), rng.randint(0, N - 1)
                if u == v:
                    continue
                u, v = min(u, v), max(u, v)
                if (u, v) in current_edges:
                    continue
                if not check_c6_fast(u, v, current_adj):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    current_count += 1
                    improved = True
                    no_improve = 0
                    break
            if improved:
                continue

            no_improve += 1
            if no_improve > max_no_improve:
                break
            if current_count == 0:
                break

            # --- Move 3 (preferred for net gain): 2-add-1-remove ---
            e_remove = rng.choice(list(current_edges))
            u1, v1 = e_remove

            current_adj[u1] &= ~(1 << v1)
            current_adj[v1] &= ~(1 << u1)
            current_edges.discard(e_remove)

            added_edges = []
            for _ in range(60):
                u2, v2 = rng.randint(0, N - 1), rng.randint(0, N - 1)
                if u2 == v2:
                    continue
                u2, v2 = min(u2, v2), max(u2, v2)
                if (u2, v2) == (u1, v1) or (u2, v2) in current_edges:
                    continue
                if not check_c6_fast(u2, v2, current_adj):
                    current_adj[u2] |= (1 << v2)
                    current_adj[v2] |= (1 << u2)
                    current_edges.add((u2, v2))
                    added_edges.append((u2, v2))
                    if len(added_edges) == 2:
                        break

            if len(added_edges) >= 2:
                current_count += len(added_edges) - 1
                no_improve = 0
                improved = True
            else:
                # revert
                for ue, ve in added_edges:
                    current_adj[ue] &= ~(1 << ve)
                    current_adj[ve] &= ~(1 << ue)
                    current_edges.discard((ue, ve))
                current_adj[u1] |= (1 << v1)
                current_adj[v1] |= (1 << u1)
                current_edges.add(e_remove)

            if improved:
                continue

            # --- Move 2: 1-swap (remove e1, add e2), only if it enables a later add ---
            for _ in range(30):
                e_rm = rng.choice(list(current_edges))
                ru, rv = e_rm
                u2, v2 = rng.randint(0, N - 1), rng.randint(0, N - 1)
                if u2 == v2:
                    continue
                u2, v2 = min(u2, v2), max(u2, v2)
                if (u2, v2) == (ru, rv) or (u2, v2) in current_edges:
                    continue

                current_adj[ru] &= ~(1 << rv)
                current_adj[rv] &= ~(1 << ru)
                current_edges.discard(e_rm)

                if not check_c6_fast(u2, v2, current_adj):
                    current_adj[u2] |= (1 << v2)
                    current_adj[v2] |= (1 << u2)
                    current_edges.add((u2, v2))

                    # try to add one more to net gain
                    got = False
                    for _ in range(12):
                        u3, v3 = rng.randint(0, N - 1), rng.randint(0, N - 1)
                        if u3 == v3:
                            continue
                        u3, v3 = min(u3, v3), max(u3, v3)
                        if (u3, v3) in current_edges:
                            continue
                        if not check_c6_fast(u3, v3, current_adj):
                            current_adj[u3] |= (1 << v3)
                            current_adj[v3] |= (1 << u3)
                            current_edges.add((u3, v3))
                            current_count += 1
                            got = True
                            no_improve = 0
                            break
                    if got:
                        break
                    else:
                        # revert the swap
                        current_adj[u2] &= ~(1 << v2)
                        current_adj[v2] &= ~(1 << u2)
                        current_edges.discard((u2, v2))
                        current_adj[ru] |= (1 << rv)
                        current_adj[rv] |= (1 << ru)
                        current_edges.add(e_rm)
                else:
                    current_adj[ru] |= (1 << rv)
                    current_adj[rv] |= (1 << ru)
                    current_edges.add(e_rm)

        return current_edges, current_adj, current_count

    def final_exhaustive_add(best_edges, best_adj):
        """Add every non-edge that doesn't create a C6, using forbidden set."""
        adj = list(best_adj)
        edges = set(best_edges)
        count = len(edges)

        # Precompute forbidden set once, then re-check cheaply.
        # Since adding edges only makes more pairs forbidden, we recompute
        # lazily: try each candidate, and if it's addable, recompute forbidden.
        # To keep it fast, iterate candidates and use check_c6_fast directly
        # but skip pairs already known forbidden via a cached set that we
        # refresh only when we actually add an edge.
        forbidden = compute_forbidden(adj)

        for u in range(N):
            for v in range(u + 1, N):
                if (u, v) in edges:
                    continue
                if (u, v) in forbidden:
                    continue
                if check_c6_fast(u, v, adj):
                    forbidden.add((u, v))
                    continue
                # addable
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
                count += 1
                # adding an edge can only add to forbidden; refresh cheaply
                forbidden = compute_forbidden(adj)
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

    # ---- Phase 3: local search on the best so far (multiple seeds) ----
    if best_edges and time.time() < deadline:
        adj = [0] * N
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        for ls_seed in range(15):
            if time.time() >= deadline:
                break
            rng = random.Random(100000 + ls_seed * 31)
            new_edges, new_adj, new_count = local_search(
                best_edges, adj, rng, max_iters=2500)
            if new_count > best_count:
                best_count = new_count
                best_edges = list(new_edges)
                best_adj = list(new_adj)
                adj = list(new_adj)

    # ---- Phase 4: final exhaustive add (accelerated) ----
    if best_edges and time.time() < deadline:
        new_edges, new_adj, new_count = final_exhaustive_add(best_edges, best_adj)
        if new_count > best_count:
            best_count = new_count
            best_edges = list(new_edges)
            best_adj = list(new_adj)

    # ---- Phase 5: one more local-search polish if time remains ----
    if best_edges and time.time() < deadline:
        adj = [0] * N
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        rng = random.Random(987654)
        new_edges, new_adj, new_count = local_search(
            best_edges, adj, rng, max_iters=2000)
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
