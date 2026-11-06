import time
import random
import numpy as np

def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit

    def recompute(adj):
        """Compute N1, N2, N4 bitsets for C6 checking."""
        N1 = adj[:]
        N2 = [0] * n
        for i in range(n):
            mask = 0
            x = N1[i]
            while x:
                lsb = x & -x
                j = lsb.bit_length() - 1
                mask |= N1[j]
                x ^= lsb
            N2[i] = mask

        N4 = [0] * n
        for i in range(n):
            mask = 0
            x = N2[i]
            while x:
                lsb = x & -x
                j = lsb.bit_length() - 1
                mask |= N2[j]
                x ^= lsb
            N4[i] = mask
        return N1, N2, N4

    def is_safe(u, v, N1, N4):
        """Check if adding edge (u,v) creates no C6."""
        return (N1[u] & N4[v]) == 0 and (N1[v] & N4[u]) == 0

    def get_edges(adj):
        edges = []
        for u in range(n):
            x = adj[u]
            while x:
                lsb = x & -x
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                x ^= lsb
        return edges

    def make_seed_adj():
        """Create adjacency for 8 disjoint K5s."""
        adj = [0] * n
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
        return adj

    def bqp_init(deadline):
        """
        BQP-inspired initialization:
        - Represent each potential edge as a binary variable x_e in {0,1}
        - Objective: maximize sum(x_e) - penalty * sum(C6_count_terms)
        - C6 penalty: for each 6-cycle, product of its 6 edge variables
        - Use spectral approximation: enforce that for non-adjacent (u,v),
          the number of length-5 paths between u and v is zero.
        
        We use a smooth relaxation: minimize -sum(x) + lambda * ||A^5||_F^2
        where A is the adjacency matrix (symmetric, zero diagonal).
        C6-free implies no 6-cycles, and 6-cycles contribute to (A^5)_{uv}
        for non-adjacent u,v.
        
        We solve via projected gradient descent on the continuous relaxation,
        then threshold to get a binary solution.
        """
        # Build the list of all possible edges
        edge_list = [(u, v) for u in range(n) for v in range(u + 1, n)]
        m_edges = len(edge_list)
        
        # Map edge index to (u, v)
        edge_idx = {}
        for i, (u, v) in enumerate(edge_list):
            edge_idx[(u, v)] = i
            edge_idx[(v, u)] = i
        
        # Precompute: for each edge e, which other edges form paths
        # For C6 detection via A^5: (A^5)_{uv} counts walks of length 5 from u to v
        # A 6-cycle u-v-w-x-y-z-u contributes to (A^5)_{uz} and (A^5)_{zu} etc.
        
        # Simpler approach: use the fact that if we have a C6-free graph,
        # then for any non-edge (u,v), N(u) and N4(v) are disjoint.
        # We'll use a penalty based on |N(u) ∩ N4(v)| for non-edges.
        
        # Continuous relaxation: x in [0,1] for each edge
        # Objective: maximize sum(x) - penalty * sum over non-edges (u,v) of |N_x(u) ∩ N4_x(v)|
        # where N_x and N4_x are computed with weighted adjacency.
        
        # This is complex, so let's use a simpler spectral penalty:
        # minimize -1^T x + lambda * tr((A_x)^6) / 6
        # where A_x is the weighted adjacency matrix.
        
        # For efficiency, we'll do a few iterations of:
        # 1. Start from a random or seed-based binary solution
        # 2. Compute gradient of the relaxed objective
        # 3. Update x via projected gradient step
        # 4. Threshold to binary
        
        best_adj = [0] * n
        best_count = 0
        
        num_iters = 50  # will be limited by deadline
        lambda_val = 10.0  # penalty weight
        
        for trial in range(3):
            if time.time() > deadline:
                break
            
            # Initialize x
            if trial == 0:
                # Start from seed
                adj0 = make_seed_adj()
                x = np.zeros(m_edges)
                for u in range(n):
                    mask = adj0[u]
                    while mask:
                        lsb = mask & -mask
                        v = lsb.bit_length() - 1
                        if u < v:
                            x[edge_idx[(u, v)]] = 1.0
                        mask ^= lsb
            elif trial == 1:
                # Random dense initialization
                x = np.random.rand(m_edges)
                x[x > 0.5] = 1.0
                x[x <= 0.5] = 0.0
            else:
                # Sparse random
                x = np.random.rand(m_edges)
                x[x > 0.3] = 1.0
                x[x <= 0.3] = 0.0
            
            # Projected gradient descent
            lr = 0.05
            for it in range(num_iters):
                if time.time() > deadline:
                    break
                
                # Build weighted adjacency matrix
                A = np.zeros((n, n))
                for i, (u, v) in enumerate(edge_list):
                    A[u, v] = x[i]
                    A[v, u] = x[i]
                
                # Compute A^2, A^4, A^6 for spectral penalty
                A2 = A @ A
                A4 = A2 @ A2
                A6 = A4 @ A2
                
                # Gradient of -sum(x) + lambda * tr(A^6)/6
                # d/dx_i [tr(A^6)] = 6 * sum of paths of length 5 through edge i
                # More precisely: d/dA_{uv} tr(A^6) = 6 * (A^5)_{vu}
                # So gradient w.r.t. x_i (for edge (u,v)) = 6 * [(A^5)_{vu} + (A^5)_{uv}] / 2 * 2
                # = 6 * (A^5)[u,v] (by symmetry)
                
                A5 = A4 @ A
                
                grad = np.ones(m_edges)  # gradient of -sum(x) is -1, so we add 1
                for i, (u, v) in enumerate(edge_list):
                    # d/dx_i tr(A^6) = 6 * (A^5)[u,v]
                    grad[i] = -1.0 + lambda_val * 6.0 * A5[u, v]
                
                # Projected gradient step
                x = x - lr * grad
                x = np.clip(x, 0.0, 1.0)
            
            # Threshold to binary
            x_bin = (x > 0.5).astype(int)
            
            # Convert to adjacency
            adj = [0] * n
            for i, (u, v) in enumerate(edge_list):
                if x_bin[i]:
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
            
            # Verify C6-free and count edges
            N1, N2, N4 = recompute(adj)
            valid = True
            edge_count = 0
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u] & (1 << v):
                        edge_count += 1
                    else:
                        # Check C6 condition
                        if (N1[u] & N4[v]) != 0 or (N1[v] & N4[u]) != 0:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid and edge_count > best_count:
                best_count = edge_count
                best_adj = adj[:]
        
        if best_count > 0:
            return best_adj
        
        # Fallback: return seed
        return make_seed_adj()

    def greedy_add(adj, N1, N2, N4, order, deadline):
        """Try to add edges in given order, updating N1/N2/N4 incrementally."""
        added = 0
        for u, v in order:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            if is_safe(u, v, N1, N4):
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                added += 1
                N1, N2, N4 = recompute(adj)
        return adj, N1, N2, N4, added

    def local_search(adj, N1, N2, N4, deadline):
        """Try to improve by removing one edge and re-adding others."""
        edges = get_edges(adj)
        best_adj = adj[:]
        best_N1, best_N2, best_N4 = N1[:], N2[:], N4[:]
        best_count = len(edges)

        # Try removing each edge and see if we can add more
        for idx, (u, v) in enumerate(edges):
            if time.time() > deadline:
                break

            # Remove edge (u, v)
            test_adj = adj[:]
            test_adj[u] &= ~(1 << v)
            test_adj[v] &= ~(1 << u)

            # Recompute
            tN1, tN2, tN4 = recompute(test_adj)

            # Try to greedily add back edges (including possibly (u,v) and others)
            candidates = []
            for a in range(n):
                for b in range(a + 1, n):
                    if not (test_adj[a] & (1 << b)):
                        candidates.append((a, b))
            random.shuffle(candidates)

            test_adj, tN1, tN2, tN4, added = greedy_add(test_adj, tN1, tN2, tN4, candidates, deadline)
            new_count = len(get_edges(test_adj))

            if new_count > best_count:
                best_count = new_count
                best_adj = test_adj
                best_N1, best_N2, best_N4 = tN1, tN2, tN4

        return best_adj, best_N1, best_N2, best_N4, best_count

    best_edges = []
    best_count = 0

    # Build candidate edge list
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]

    # First, try BQP-inspired initialization
    bqp_adj = bqp_init(deadline)
    bqp_edges = get_edges(bqp_adj)
    if len(bqp_edges) > best_count:
        best_count = len(bqp_edges)
        best_edges = bqp_edges

    # Multiple strategies
    while time.time() < deadline:
        strategy = random.random()

        if strategy < 0.25:
            # Strategy 1: Greedy from seed with random order
            adj = make_seed_adj()
            N1, N2, N4 = recompute(adj)

            candidates = [(u, v) for u, v in all_pairs if not (adj[u] & (1 << v))]
            random.shuffle(candidates)

            adj, N1, N2, N4, _ = greedy_add(adj, N1, N2, N4, candidates, deadline)
            edges = get_edges(adj)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges

        elif strategy < 0.5:
            # Strategy 2: Greedy from BQP result with random order
            adj = bqp_adj[:]
            N1, N2, N4 = recompute(adj)

            candidates = [(u, v) for u, v in all_pairs if not (adj[u] & (1 << v))]
            random.shuffle(candidates)

            adj, N1, N2, N4, _ = greedy_add(adj, N1, N2, N4, candidates, deadline)
            edges = get_edges(adj)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges

        elif strategy < 0.75:
            # Strategy 3: Greedy from seed + local search
            adj = make_seed_adj()
            N1, N2, N4 = recompute(adj)

            candidates = [(u, v) for u, v in all_pairs if not (adj[u] & (1 << v))]
            random.shuffle(candidates)

            adj, N1, N2, N4, _ = greedy_add(adj, N1, N2, N4, candidates, deadline)

            # Local search
            adj, N1, N2, N4, count = local_search(adj, N1, N2, N4, deadline)
            edges = get_edges(adj)
            if count > best_count:
                best_count = count
                best_edges = edges

        else:
            # Strategy 4: Perturbed BQP result
            adj = bqp_adj[:]
            # Remove 2-4 random edges
            edges_in_adj = get_edges(adj)
            if len(edges_in_adj) > 4:
                to_remove = random.sample(edges_in_adj, min(4, len(edges_in_adj)))
                for u, v in to_remove:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)

            N1, N2, N4 = recompute(adj)

            candidates = [(u, v) for u, v in all_pairs if not (adj[u] & (1 << v))]
            random.shuffle(candidates)

            adj, N1, N2, N4, _ = greedy_add(adj, N1, N2, N4, candidates, deadline)
            edges = get_edges(adj)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges

    return best_edges
