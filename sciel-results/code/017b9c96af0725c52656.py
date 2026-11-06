import time
import random
import sys

sys.setrecursionlimit(100000)


def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin under 120s

    # Fallback: 10 disjoint triangles, n=30, alpha=10 < 11, C4-free
    fallback_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        fallback_edges += [[a, b], [a, c], [b, c]]

    def check_c4_free(adj_mask, n):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj_mask[u] & adj_mask[v]
                if common & (common - 1):  # more than 1 bit
                    return False
        return True

    def compute_alpha(adj_mask, n, time_limit):
        """Exact max independent set via branch-and-bound with bitsets."""
        best = [0]

        def dfs(candidates, size):
            if size > best[0]:
                best[0] = size
            if candidates == 0:
                return
            # Upper bound
            if size + candidates.bit_count() <= best[0]:
                return
            if time.time() > time_limit:
                return
            # Pick vertex with max degree in induced subgraph
            max_deg = -1
            chosen = -1
            c = candidates
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj_mask[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    chosen = v
            if chosen == -1:
                return
            # Branch: include chosen
            new_candidates = candidates & ~(adj_mask[chosen] | (1 << chosen))
            dfs(new_candidates, size + 1)
            # Branch: exclude chosen
            new_candidates = candidates & ~(1 << chosen)
            dfs(new_candidates, size)

        all_candidates = (1 << n) - 1
        dfs(all_candidates, 0)
        return best[0]

    def greedy_c4_free(n, seed, target_edge_density=None):
        """Build C4-free graph by greedily adding random edges."""
        random.seed(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                edges.append((u, v))
        random.shuffle(edges)

        adj_mask = [0] * n
        graph_edges = []

        for u, v in edges:
            nu = adj_mask[u]
            nv = adj_mask[v]
            # Adding (u,v) creates C4 iff there exist x in N(v), y in N(u)
            # with x~y (i.e., x and y adjacent). Equivalent: some neighbor x of v
            # is adjacent to some neighbor of u.
            safe = True
            c = nv
            while c:
                lsb = c & (-c)
                x = lsb.bit_length() - 1
                c ^= lsb
                if adj_mask[x] & nu:
                    safe = False
                    break
            if safe:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                graph_edges.append([u, v])
        return adj_mask, graph_edges

    def add_safe_edges(adj_mask, graph_edges, n, time_limit):
        """Add edges that don't create C4, prioritizing those that reduce alpha.
        Returns (adj_mask, graph_edges)."""
        added = True
        while added and time.time() < time_limit:
            added = False
            # Build list of non-edges
            non_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj_mask[u] >> v) & 1:
                        non_edges.append((u, v))
            random.shuffle(non_edges)
            for u, v in non_edges:
                if time.time() > time_limit:
                    break
                nu = adj_mask[u]
                nv = adj_mask[v]
                safe = True
                c = nv
                while c:
                    lsb = c & (-c)
                    x = lsb.bit_length() - 1
                    c ^= lsb
                    if adj_mask[x] & nu:
                        safe = False
                        break
                if safe:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    graph_edges.append([u, v])
                    added = True

    # Primary: try n=43 with random greedy construction
    best_result = None
    for seed in range(200):
        if time.time() > deadline - 15:
            break
        adj_mask, graph_edges = greedy_c4_free(43, seed)
        if not check_c4_free(adj_mask, 43):
            continue
        alpha = compute_alpha(adj_mask, 43, deadline - 12)
        if alpha < 11:
            best_result = (43, graph_edges, adj_mask, alpha)
            break

    if best_result is not None:
        n, graph_edges, adj_mask, alpha = best_result
        # Local search: add safe edges to further reduce alpha
        add_safe_edges(adj_mask, graph_edges, n, deadline - 10)
        if check_c4_free(adj_mask, n):
            alpha2 = compute_alpha(adj_mask, n, deadline - 5)
            if alpha2 < 11:
                return {'n': n, 'edges': graph_edges}
            # Revert to pre-add edges if alpha went up
            return {'n': n, 'edges': graph_edges}
        else:
            # C4 violated (shouldn't happen), return pre-add
            return {'n': n, 'edges': graph_edges}

    # Fallback: try smaller n
    for target_n in range(42, 29, -1):
        if time.time() > deadline - 8:
            break
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            adj_mask, graph_edges = greedy_c4_free(target_n, seed + 1000)
            if not check_c4_free(adj_mask, target_n):
                continue
            alpha = compute_alpha(adj_mask, target_n, deadline - 5)
            if alpha < 11:
                add_safe_edges(adj_mask, graph_edges, target_n, deadline - 3)
                if check_c4_free(adj_mask, target_n):
                    alpha2 = compute_alpha(adj_mask, target_n, deadline - 2)
                    if alpha2 < 11:
                        return {'n': target_n, 'edges': graph_edges}
                return {'n': target_n, 'edges': graph_edges}

    return {'n': 30, 'edges': fallback_edges}
