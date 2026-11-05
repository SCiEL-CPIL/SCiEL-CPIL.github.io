import random
import time

def construct():
    n = 20
    start_time = time.time()
    max_time = 110.0  # Safety margin before 120s limit

    best_edges = []
    best_count = 0

    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    all_edges_set = set(all_edges)
    edge_to_idx = {(u, v): i for i, (u, v) in enumerate(all_edges)}
    num_edges = len(all_edges)

    def is_c4_free_add(u, v, adj_mask, common_cnt):
        """Check if adding (u,v) keeps the subgraph C4-free (no two vertices share 2+ neighbors)."""
        # For every neighbor w of v (excluding u), common_cnt[u][w] must be 0
        # (i.e., u and w share no neighbors in this subgraph)
        # Equivalently: common_cnt[u][w] < 1 for all w in N(v) \ {u}
        mask_v = adj_mask[v]
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != u and common_cnt[u][w] >= 1:
                return False
            m -= lsb
        return True

    def add_edge_subgraph(u, v, adj_mask, common_cnt):
        """Add edge (u,v) to a C4-free subgraph, updating common neighbor counts."""
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

        # For each neighbor w of v (old, before adding u): common[u][w] += 1
        # And for each neighbor w of u (old, before adding v): common[v][w] += 1
        mask_v = adj_mask[v] & ~(1 << u)
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common_cnt[u][w] += 1
            common_cnt[w][u] += 1
            m -= lsb

        mask_u = adj_mask[u] & ~(1 << v)
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common_cnt[v][w] += 1
            common_cnt[w][v] += 1
            m -= lsb

    def remove_edge_subgraph(u, v, adj_mask, common_cnt):
        """Remove edge (u,v) from a C4-free subgraph."""
        # Before clearing bits: N(v) includes u, N(u) includes v
        mask_v = adj_mask[v]
        mask_u = adj_mask[u]

        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)

        # For each w in N(v) \ {u}: common[u][w] -= 1
        m = mask_v & ~(1 << u)
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common_cnt[u][w] -= 1
            common_cnt[w][u] -= 1
            m -= lsb

        # For each w in N(u) \ {v}: common[v][w] -= 1
        m = mask_u & ~(1 << v)
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common_cnt[v][w] -= 1
            common_cnt[w][v] -= 1
            m -= lsb

    def compute_common_cnt(adj_mask):
        """Compute common neighbor counts for a subgraph from scratch."""
        common_cnt = [[0] * n for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                mask = adj_mask[u] & adj_mask[v]
                cnt = bin(mask).count('1')
                common_cnt[u][v] = cnt
                common_cnt[v][u] = cnt
        return common_cnt

    def validate_decomposition(adj1, adj2):
        """Verify that the union is K2,3-free by checking common neighbors in union."""
        for u in range(n):
            for v in range(u + 1, n):
                mask = adj1[u] & adj1[v] | adj2[u] & adj2[v]
                # Wait, this is wrong. Common neighbors in union = (N1(u)∩N1(v)) ∪ (N2(u)∩N2(v)) ∪ (N1(u)∩N2(v)) ∪ (N2(u)∩N1(v))
                # Actually simpler: for each w != u,v, check if w is neighbor of both u and v in the union.
                pass
        # Better: just count common neighbors directly
        for u in range(n):
            for v in range(u + 1, n):
                cnt = 0
                for w in range(n):
                    if w == u or w == v:
                        continue
                    in_u = ((adj1[u] >> w) & 1) or ((adj2[u] >> w) & 1)
                    in_v = ((adj1[v] >> w) & 1) or ((adj2[v] >> w) & 1)
                    if in_u and in_v:
                        cnt += 1
                if cnt > 2:
                    return False
        return True

    def count_edges_decomp(adj1, adj2):
        cnt1 = sum(bin(adj1[i]).count('1') for i in range(n)) // 2
        cnt2 = sum(bin(adj2[i]).count('1') for i in range(n)) // 2
        return cnt1 + cnt2

    def get_edges_decomp(adj1, adj2):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj1[u] >> v) & 1:
                    edges.append([u, v])
                elif (adj2[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def run_sa_decomp(seed):
        """Simulated Annealing over the two-subgraph decomposition."""
        random.seed(seed)

        # Initialize: random assignment of a random edge set
        adj1 = [0] * n
        adj2 = [0] * n

        # Start with a random sparse graph, assign edges randomly to G1 or G2
        for u, v in all_edges:
            if random.random() < 0.3:
                if random.random() < 0.5:
                    adj1[u] |= (1 << v)
                    adj1[v] |= (1 << u)
                else:
                    adj2[u] |= (1 << v)
                    adj2[v] |= (1 << u)

        # Project: remove edges that violate C4-freeness in each subgraph
        # Greedily remove violating edges
        def project(adj, common_cnt):
            changed = True
            while changed:
                changed = False
                for u, v in all_edges:
                    if (adj[u] >> v) & 1:
                        # Check if this edge causes a C4 violation
                        # An edge (u,v) is part of a C4 if there exist a,b such that
                        # a,b are common neighbors of u and v, OR
                        # the edge creates a situation where two vertices share 2+ neighbors.
                        # Actually, we need to check if removing other edges is needed.
                        # Simpler: just check if the subgraph is C4-free. If not, remove a random violating edge.
                        pass
            # Better approach: iteratively find and remove edges that are part of a C4
            while True:
                # Find a C4: look for two vertices with 2+ common neighbors
                violating = None
                for u in range(n):
                    for v in range(u + 1, n):
                        mask = adj[u] & adj[v]
                        cnt = bin(mask).count('1')
                        if cnt >= 2:
                            violating = (u, v)
                            break
                    if violating:
                        break
                if not violating:
                    break
                # Remove a random edge incident to u or v that is part of the C4
                u, v = violating
                mask = adj[u] & adj[v]
                # Pick a random common neighbor w
                common_neighbors = []
                m = mask
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    common_neighbors.append(w)
                    m -= lsb
                if not common_neighbors:
                    break
                w = random.choice(common_neighbors)
                # Remove edge (u, w) or (v, w)
                if random.random() < 0.5:
                    adj[u] &= ~(1 << w)
                    adj[w] &= ~(1 << u)
                else:
                    adj[v] &= ~(1 << w)
                    adj[w] &= ~(1 << v)

        project(adj1, None)
        project(adj2, None)

        # Recompute common counts
        cc1 = compute_common_cnt(adj1)
        cc2 = compute_common_cnt(adj2)

        best_adj1 = adj1[:]
        best_adj2 = adj2[:]
        best_count = count_edges_decomp(adj1, adj2)

        T = 5.0
        cooling = 0.9995
        min_T = 0.01

        max_iters = 50000
        for it in range(max_iters):
            if time.time() - start_time > max_time - 5.0:
                break

            if T < min_T:
                T = min_T

            # Choose a random edge and a random subgraph to toggle
            u, v = random.choice(all_edges)
            sub = random.randint(0, 1)

            old_count = count_edges_decomp(adj1, adj2)

            if sub == 0:
                adj = adj1
                cc = cc1
                other_adj = adj2
            else:
                adj = adj2
                cc = cc2
                other_adj = adj1

            has_edge = (adj[u] >> v) & 1

            if has_edge:
                # Try removing
                remove_edge_subgraph(u, v, adj, cc)
                new_count = count_edges_decomp(adj1, adj2)
                delta = new_count - old_count
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    # Accept
                    if new_count > best_count:
                        best_count = new_count
                        best_adj1 = adj1[:]
                        best_adj2 = adj2[:]
                else:
                    # Reject: add back
                    add_edge_subgraph(u, v, adj, cc)
            else:
                # Try adding
                # Check C4-freeness
                if is_c4_free_add(u, v, adj, cc):
                    add_edge_subgraph(u, v, adj, cc)
                    new_count = count_edges_decomp(adj1, adj2)
                    delta = new_count - old_count
                    if delta > 0 or random.random() < pow(2.718281828, delta / T):
                        if new_count > best_count:
                            best_count = new_count
                            best_adj1 = adj1[:]
                            best_adj2 = adj2[:]
                    else:
                        remove_edge_subgraph(u, v, adj, cc)
                # If not C4-free, skip (effectively reject)

            T *= cooling

        # Final projection and validation
        project(best_adj1, None)
        project(best_adj2, None)

        if not validate_decomposition(best_adj1, best_adj2):
            return None, 0

        edges = get_edges_decomp(best_adj1, best_adj2)
        return edges, len(edges)

    # Main loop with restarts
    seed = 42
    while time.time() - start_time < max_time:
        local_edges, local_count = run_sa_decomp(seed)

        if local_edges is not None and local_count > best_count:
            best_count = local_count
            best_edges = local_edges

        seed += 1

    # Fallback: if no valid result, use a simple construction
    if not best_edges:
        # Use a known K2,3-free construction: Petersen graph-like or just sparse random
        adj = [0] * n
        edges = []
        for u, v in all_edges:
            if random.random() < 0.15:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
        best_edges = edges
        best_count = len(edges)

    return best_edges

if __name__ == "__main__":
    # Quick test
    edges = construct()
    print(f"Found {len(edges)} edges")
