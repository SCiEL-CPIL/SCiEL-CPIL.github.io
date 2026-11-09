import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    squares = {x * x % 17 for x in range(1, 17)}
    edges_17 = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    best_n = 17
    best_edges = edges_17

    adj17 = np.zeros((17, 17), dtype=np.int8)
    for u, v in edges_17:
        adj17[u, v] = 1
        adj17[v, u] = 1

    def build_masks(n, adj):
        masks = [0] * n
        for u in range(n):
            for v in range(n):
                if adj[u, v]:
                    masks[u] |= (1 << v)
        return masks

    def count_k4_delta(n, adj, u, v, old_state):
        """Count K4s that change when flipping edge (u,v).
        Returns the delta in K4 count (positive means K4s added)."""
        # We need to count pairs (a,b) such that {u,v,a,b} forms a K4.
        # This requires: u-v edge, u-a, u-b, v-a, v-b, a-b edges all present.
        # The only edge whose status changes is (u,v).
        # So K4s are created/destroyed exactly when all other 5 edges are present.
        # We check using the CURRENT adj (after flip), but we need the state of
        # the 5 other edges which don't change.
        
        # For adding edge (u,v): K4s created = number of (a,b) pairs where
        #   a,b != u,v; a-b edge exists; u-a, u-b, v-a, v-b all exist.
        # For removing edge (u,v): K4s destroyed = same count.
        
        # Use masks from current adj (after flip), but exclude u-v check
        # since we know its state changed.
        count = 0
        # a and b must both be neighbors of u (excluding v) and neighbors of v (excluding u)
        # and a-b must be an edge
        for a in range(n):
            if a == u or a == v:
                continue
            if not (adj[u, a] and adj[v, a]):
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if not (adj[u, b] and adj[v, b]):
                    continue
                if adj[a, b]:
                    count += 1
        return count

    def count_ind6_delta(n, adj, u, v, old_state):
        """Count independent 6-sets that change when flipping edge (u,v).
        Returns the delta in independent-6-set count (positive means more ind6 sets)."""
        # When we ADD edge (u,v): independent sets containing both u and v are destroyed.
        #   Delta = - (number of independent 4-sets from {u,v} plus 2 others that are
        #   independent from each other and from u,v)
        # When we REMOVE edge (u,v): independent sets containing both u and v are created.
        #   Delta = + (same count)
        
        # Count 4-sets {u,v,a,b} where all 6 pairs are non-edges.
        # The only pair whose status changes is (u,v).
        # So we need: a-b non-edge, u-a non-edge, u-b non-edge, v-a non-edge, v-b non-edge.
        count = 0
        for a in range(n):
            if a == u or a == v:
                continue
            if adj[u, a] or adj[v, a]:
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if adj[u, b] or adj[v, b]:
                    continue
                if not adj[a, b]:
                    count += 1
        return count

    def cost_full(n, adj):
        """Full cost: count all K4s and independent 6-sets."""
        masks = build_masks(n, adj)
        k4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (masks[u] & (1 << v)):
                    continue
                common = masks[u] & masks[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if masks[wi] & (1 << xi):
                            k4_count += 1

        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(masks[u] | (1 << u))

        ind6_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (comp_masks[u] & (1 << v)):
                    continue
                common = comp_masks[u] & comp_masks[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if comp_masks[wi] & (1 << xi):
                            common2 = comp_masks[u] & comp_masks[v] & comp_masks[wi] & comp_masks[xi]
                            common2 &= ~((1 << (xi + 1)) - 1)
                            y_mask = common2
                            while y_mask:
                                lsb_y = y_mask & -y_mask
                                yi = lsb_y.bit_length() - 1
                                y_mask ^= lsb_y
                                z_mask = common2 & ~((1 << (yi + 1)) - 1)
                                while z_mask:
                                    lsb_z = z_mask & -z_mask
                                    zi = lsb_z.bit_length() - 1
                                    z_mask ^= lsb_z
                                    if comp_masks[yi] & (1 << zi):
                                        ind6_count += 1

        return k4_count + ind6_count

    def is_valid(n, adj):
        return cost_full(n, adj) == 0

    def spectral_init(n, rng):
        """Generate adjacency via spectral method using Paley-type construction.
        Uses quadratic residue structure to create graphs with good K4/Ind6 properties."""
        # Try multiple approaches
        approach = rng.integers(0, 3)
        
        if approach == 0:
            # Paley-like: use modular arithmetic
            # Find smallest prime p >= n
            p = n
            while True:
                is_prime = True
                for d in range(2, int(p**0.5) + 1):
                    if p % d == 0:
                        is_prime = False
                        break
                if is_prime:
                    break
                p += 1
            # Quadratic residues mod p
            squares = set()
            for x in range(1, p):
                squares.add(x * x % p)
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i + 1, n):
                    diff = (j - i) % p
                    if diff in squares:
                        adj[i, j] = 1
                        adj[j, i] = 1
            return adj
        elif approach == 1:
            # Random symmetric matrix with block structure, eigenvector sign binarization
            A = rng.standard_normal((n, n))
            A = (A + A.T) / 2.0
            # Add block structure for better spectral properties
            num_blocks = rng.integers(2, 5)
            block_size = n // num_blocks
            for i in range(n):
                for j in range(n):
                    if (i // max(1, block_size)) == (j // max(1, block_size)):
                        A[i, j] += 1.0
                    elif i != j:
                        A[i, j] -= 0.3
            eigvals, eigvecs = np.linalg.eigh(A)
            # Use top eigenvector signs
            top_vec = eigvecs[:, -1]
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i + 1, n):
                    # Edge if both have same sign in top eigenvector
                    if top_vec[i] * top_vec[j] > 0:
                        adj[i, j] = 1
                        adj[j, i] = 1
            return adj
        else:
            # Complete bipartite-like with random perturbation
            adj = np.zeros((n, n), dtype=np.int8)
            mid = n // 2
            for i in range(mid):
                for j in range(mid, n):
                    if rng.random() < 0.5:
                        adj[i, j] = 1
                        adj[j, i] = 1
            # Add some intra-part edges
            for i in range(mid):
                for j in range(i + 1, mid):
                    if rng.random() < 0.2:
                        adj[i, j] = 1
                        adj[j, i] = 1
            for i in range(mid, n):
                for j in range(i + 1, n):
                    if rng.random() < 0.2:
                        adj[i, j] = 1
                        adj[j, i] = 1
            return adj

    def try_find_graph(n, time_budget, seed_adj=None):
        end_time = time.time() + time_budget

        if seed_adj is not None and seed_adj.shape == (n, n):
            adj = seed_adj.copy()
        else:
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i, j] = 1
                        adj[j, i] = 1

        current_cost = cost_full(n, adj)
        if current_cost == 0:
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            return edges

        T = 3.0
        min_T = 0.01
        cooling = 0.999
        no_improve = 0
        max_no_improve = 5000
        step = 0

        while time.time() < end_time and no_improve < max_no_improve:
            step += 1
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue

            old_state = adj[u, v]
            adj[u, v] = 1 - old_state
            adj[v, u] = 1 - old_state

            # Incremental cost update
            if old_state == 0:
                # Adding edge (u,v): increases K4 count, decreases ind6 count
                k4_delta = count_k4_delta(n, adj, u, v, old_state)
                ind6_delta = -count_ind6_delta(n, adj, u, v, old_state)
            else:
                # Removing edge (u,v): decreases K4 count, increases ind6 count
                k4_delta = -count_k4_delta(n, adj, u, v, old_state)
                ind6_delta = count_ind6_delta(n, adj, u, v, old_state)

            new_cost = current_cost + k4_delta + ind6_delta

            if new_cost < current_cost:
                current_cost = new_cost
                no_improve = 0
                if current_cost <= 0:
                    # Verify with full check
                    if is_valid(n, adj):
                        edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
                        return edges
            else:
                delta = new_cost - current_cost
                T = max(min_T, T * cooling)
                if random.random() < np.exp(-delta / T):
                    current_cost = new_cost
                    no_improve += 1
                else:
                    adj[u, v] = old_state
                    adj[v, u] = old_state
                    no_improve += 1

        if is_valid(n, adj):
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            return edges

        return None

    # Try to find graphs from n=18 upward
    for n in range(18, 36):
        if time.time() > deadline:
            break

        remaining = deadline - time.time()
        if remaining < 5.0:
            break

        # Allocate time based on n
        if n <= 22:
            budget = min(remaining * 0.35, 10.0)
        elif n <= 28:
            budget = min(remaining * 0.30, 8.0)
        else:
            budget = min(remaining * 0.25, 6.0)

        found = None
        num_restarts = 6
        per_restart = budget / num_restarts

        for restart in range(num_restarts):
            if time.time() > deadline:
                break

            if restart == 0:
                # Spectral/Paley initialization
                rng = np.random.default_rng(random.randint(0, 2**31))
                seed_adj = spectral_init(n, rng)
                found = try_find_graph(n, per_restart, seed_adj=seed_adj)
            elif restart == 1:
                # Another spectral with different params
                rng = np.random.default_rng(random.randint(0, 2**31))
                seed_adj = spectral_init(n, rng)
                found = try_find_graph(n, per_restart, seed_adj=seed_adj)
            elif restart == 2:
                # Extend Paley(17) if n >= 18
                if n >= 18:
                    seed_adj = np.zeros((n, n), dtype=np.int8)
                    seed_adj[:17, :17] = adj17
                    for i in range(17, n):
                        for j in range(17):
                            if random.random() < 0.5:
                                seed_adj[i, j] = 1
                                seed_adj[j, i] = 1
                    for i in range(17, n):
                        for j in range(i + 1, n):
                            if random.random() < 0.5:
                                seed_adj[i, j] = 1
                                seed_adj[j, i] = 1
                    found = try_find_graph(n, per_restart, seed_adj=seed_adj)
                else:
                    found = try_find_graph(n, per_restart)
            else:
                # Pure random
                found = try_find_graph(n, per_restart)

            if found is not None:
                break

        if found is not None:
            best_n = n
            best_edges = found
            if n >= 35:
                break

    return {'n': best_n, 'edges': best_edges}
