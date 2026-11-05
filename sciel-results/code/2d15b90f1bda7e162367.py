import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n = 18

    def is_independent(mask, adj):
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if adj[v] & (mask ^ lsb):
                return False
            m ^= lsb
        return True

    def can_add_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            return True
        common = adj[u] & adj[v]
        return is_independent(common, adj)

    def greedy_tf_size(adj):
        """Fast greedy: repeatedly pick vertex with min degree in remaining subgraph."""
        size = 0
        remaining = (1 << n) - 1
        while remaining:
            best_v = -1
            best_deg = n + 1
            m = remaining
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = bin(adj[v] & remaining).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                m ^= lsb
            size += 1
            remaining &= ~adj[best_v]
            remaining &= ~(1 << best_v)
        return size

    def exact_tf_size(adj, time_limit=None):
        """Exact branch-and-bound for maximum triangle-free induced subgraph size."""
        best = [0]
        full_mask = (1 << n) - 1

        # Greedy lower bound
        best[0] = greedy_tf_size(adj)

        def bnb(S_mask, C_mask):
            if time_limit and time.time() > time_limit:
                return
            sz = bin(S_mask).count('1')
            if sz + bin(C_mask).count('1') <= best[0]:
                return
            if not C_mask:
                if sz > best[0]:
                    best[0] = sz
                return

            # Pick vertex with max degree in C
            max_deg = -1
            chosen_v = -1
            c = C_mask
            while c:
                lsb = c & -c
                vid = lsb.bit_length() - 1
                deg_in_c = bin(adj[vid] & C_mask).count('1')
                if deg_in_c > max_deg:
                    max_deg = deg_in_c
                    chosen_v = vid
                c ^= lsb

            v = chosen_v
            v_bit = 1 << v

            # Branch: exclude v first
            bnb(S_mask, C_mask & ~v_bit)

            # Branch: include v
            if is_independent(adj[v] & S_mask, adj):
                bnb(S_mask | v_bit, C_mask & ~v_bit)

        bnb(0, full_mask)
        return best[0]

    def get_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def generate_random_k4_free():
        adj = [0] * n
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search(adj, greedy_score, time_limit):
        """
        Smooth surrogate local search:
        Use greedy_tf_size as a fast proxy to guide edge additions.
        Only verify with exact B&B when greedy score improves significantly.
        """
        best_adj = adj[:]
        best_greedy = greedy_score
        best_exact = None
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

        while time.time() < time_limit:
            random.shuffle(all_edges)
            improved = False

            for u, v in all_edges:
                if time.time() > time_limit:
                    return best_adj, best_greedy, best_exact
                if (adj[u] >> v) & 1:
                    continue
                if not can_add_edge(adj, u, v):
                    continue

                # Try adding edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

                new_greedy = greedy_tf_size(adj)

                if new_greedy < best_greedy:
                    # Greedy proxy says improvement — verify with exact
                    new_exact = exact_tf_size(adj, time_limit=time_limit)
                    if new_exact <= best_greedy:  # exact confirms no worse
                        best_greedy = new_greedy
                        best_exact = new_exact
                        best_adj = adj[:]
                        improved = True
                    else:
                        # Exact says it's not actually better, undo
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                elif new_greedy == best_greedy:
                    # Same greedy score — might still be better exactly,
                    # but only check occasionally to save time
                    if random.random() < 0.3:
                        new_exact = exact_tf_size(adj, time_limit=time_limit)
                        if new_exact < (best_exact if best_exact is not None else best_greedy):
                            best_greedy = new_greedy
                            best_exact = new_exact
                            best_adj = adj[:]
                            improved = True
                        else:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                else:
                    # Greedy says worse — undo
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)

            if not improved:
                # Try edge removals to escape local optima
                edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n) if (adj[i] >> j) & 1]
                random.shuffle(edges_to_try)
                for u, v in edges_to_try[:10]:
                    if time.time() > time_limit:
                        return best_adj, best_greedy, best_exact
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    new_greedy = greedy_tf_size(adj)
                    if new_greedy < best_greedy:
                        new_exact = exact_tf_size(adj, time_limit=time_limit)
                        if new_exact < best_greedy:
                            best_greedy = new_greedy
                            best_exact = new_exact
                            best_adj = adj[:]
                            improved = True
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                if not improved:
                    break

        return best_adj, best_greedy, best_exact

    random.seed(42)
    best_adj = None
    best_exact_score = n

    # Phase 1: Multiple random restarts with local search
    restarts = 0
    while time.time() < deadline - 20.0:
        restarts += 1
        adj = generate_random_k4_free()
        g_score = greedy_tf_size(adj)

        # Run local search
        adj2, g2, e2 = local_search(adj, g_score, deadline - 20.0)

        if e2 is not None:
            score = e2
        else:
            score = g2

        if score < best_exact_score:
            best_exact_score = score
            best_adj = adj2[:]
            print(f"Restart {restarts}: exact={score}, greedy={g2}")

    if best_adj is None:
        best_adj = generate_random_k4_free()
        best_exact_score = exact_tf_size(best_adj)

    # Phase 2: Final polish — more local search on the best solution
    if time.time() < deadline - 5.0:
        g_score = greedy_tf_size(best_adj)
        adj3, g3, e3 = local_search(best_adj, g_score, deadline - 5.0)
        if e3 is not None and e3 < best_exact_score:
            best_exact_score = e3
            best_adj = adj3[:]
            print(f"Polish: exact={best_exact_score}")

    # Final exact verification
    if time.time() < deadline - 2.0:
        final_score = exact_tf_size(best_adj, time_limit=deadline - 1.0)
        if final_score < best_exact_score:
            best_exact_score = final_score

    edges = get_edge_list(best_adj)
    return edges
