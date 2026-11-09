import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

    # Best known seed: Paley graph on 17 vertices
    squares = {x * x % 17 for x in range(1, 17)}
    edges_17 = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    best_n = 17
    best_edges = edges_17

    def build_masks(n, adj):
        masks = [0] * n
        for u in range(n):
            for v in range(n):
                if adj[u, v]:
                    masks[u] |= (1 << v)
        return masks

    def has_k4(n, masks):
        """Check if graph contains a K4. Returns True if found."""
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
                            return True
        return False

    def has_ind6(n, masks):
        """Check if graph contains an independent 6-set. Returns True if found."""
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(masks[u] | (1 << u))

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
                                        return True
        return False

    def is_valid(n, masks):
        if has_k4(n, masks):
            return False
        if has_ind6(n, masks):
            return False
        return True

    def count_edges_in_mask(mask, masks):
        """Count edges in subgraph induced by vertices in mask."""
        count = 0
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            m ^= lsb
            count += bin(masks[u] & mask).count('1')
        return count // 2

    def count_non_edges_in_mask(mask, masks, n):
        """Count non-edges (pairs not connected by an edge) in subgraph induced by mask."""
        k = bin(mask).count('1')
        if k < 2:
            return 0
        total_pairs = k * (k - 1) // 2
        edges = count_edges_in_mask(mask, masks)
        return total_pairs - edges

    def delta_k4(n, masks, u, v, adding):
        """Delta in K4 count when toggling edge (u,v)."""
        if adding:
            # New K4s: {u,v,w,x} where w,x are common neighbors and w-x is edge
            common = masks[u] & masks[v]
            count = 0
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
                        count += 1
            return count
        else:
            # Removed K4s: same count (the K4s that used edge u-v)
            common = masks[u] & masks[v]
            count = 0
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
                        count += 1
            return -count

    def delta_ind6(n, masks, u, v, adding):
        """Delta in independent 6-set count when toggling edge (u,v)."""
        full_mask = (1 << n) - 1
        comp_u = full_mask & ~(masks[u] | (1 << u))
        comp_v = full_mask & ~(masks[v] | (1 << v))
        
        if adding:
            # Adding edge u-v removes independent sets containing both u and v
            # Need to count independent 4-sets in common non-neighbors of u and v
            common_non = comp_u & comp_v
            # Count independent 4-sets in G[common_non]
            # = count 4-cliques in complement[common_non]
            # An independent 4-set in G[S] is a 4-clique in \bar{G}[S]
            # For each pair (w,x) in common_non that are non-adjacent in G,
            # count pairs (y,z) in common_non \ {w,x} that are non-adjacent to w,x and each other
            
            count = 0
            # Iterate over pairs (w,x) in common_non
            m_w = common_non
            while m_w:
                lsb_w = m_w & -m_w
                w = lsb_w.bit_length() - 1
                m_w ^= lsb_w
                
                # x > w
                m_x = common_non & ~((1 << (w + 1)) - 1)
                while m_x:
                    lsb_x = m_x & -m_x
                    x = lsb_x.bit_length() - 1
                    m_x ^= lsb_x
                    
                    # w and x must be non-adjacent in G
                    if masks[w] & (1 << x):
                        continue
                    
                    # Find vertices in common_non that are non-adjacent to both w and x
                    # Non-neighbors of w in common_non: common_non & ~masks[w]
                    # Non-neighbors of x in common_non: common_non & ~masks[x]
                    candidates = common_non & ~(masks[w] | masks[x] | (1 << w) | (1 << x))
                    
                    # Count non-edges in G[candidates]
                    count += count_non_edges_in_mask(candidates, masks, n)
            
            return -count
        else:
            # Removing edge u-v adds independent sets containing both u and v
            common_non = comp_u & comp_v
            count = 0
            m_w = common_non
            while m_w:
                lsb_w = m_w & -m_w
                w = lsb_w.bit_length() - 1
                m_w ^= lsb_w
                
                m_x = common_non & ~((1 << (w + 1)) - 1)
                while m_x:
                    lsb_x = m_x & -m_x
                    x = lsb_x.bit_length() - 1
                    m_x ^= lsb_x
                    
                    if masks[w] & (1 << x):
                        continue
                    
                    candidates = common_non & ~(masks[w] | masks[x] | (1 << w) | (1 << x))
                    count += count_non_edges_in_mask(candidates, masks, n)
            
            return count

    def total_cost(n, masks):
        """Total number of K4s + independent 6-sets."""
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

        masks = build_masks(n, adj)
        
        # Check if already valid
        if is_valid(n, masks):
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            return edges

        # SA parameters
        T = 10.0
        min_T = 0.05
        cooling = 0.9995
        no_improve = 0
        max_no_improve = 5000
        best_cost = total_cost(n, masks)
        best_adj = adj.copy()
        best_masks = masks[:]

        while time.time() < end_time and no_improve < max_no_improve:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue

            old_state = adj[u, v]
            adding = (old_state == 0)

            # Compute delta
            dk4 = delta_k4(n, masks, u, v, adding)
            dind6 = delta_ind6(n, masks, u, v, adding)
            delta = dk4 + dind6

            # Apply toggle
            adj[u, v] = 1 - old_state
            adj[v, u] = 1 - old_state
            if adding:
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
            else:
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)

            if delta < 0:
                no_improve = 0
                # Check if valid
                if is_valid(n, masks):
                    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
                    return edges
            else:
                T = max(min_T, T * cooling)
                if random.random() < np.exp(-delta / T):
                    no_improve += 1
                else:
                    # Revert
                    adj[u, v] = old_state
                    adj[v, u] = old_state
                    if adding:
                        masks[u] &= ~(1 << v)
                        masks[v] &= ~(1 << u)
                    else:
                        masks[u] |= (1 << v)
                        masks[v] |= (1 << u)
                    no_improve += 1

        # Final check
        if is_valid(n, masks):
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            return edges

        # Try best state found during SA
        if is_valid(n, best_masks):
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if best_adj[i, j]]
            return edges

        return None

    # Algebraic seed construction using finite field geometry
    def make_finite_field_seed(n, q, p):
        """Create a seed graph using multiplicative character over GF(q^k)."""
        # For now, use a simple approach: Paley-like graph if n is a prime power
        # Otherwise, use random with bias
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.45:  # Slightly below 0.5 to reduce K4s
                    adj[i, j] = 1
                    adj[j, i] = 1
        return adj

    # Strategy: try n=35 down to 18
    for n in range(35, 17, -1):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 10.0:
            break

        # Allocate time: more time for larger n
        base_budget = min(remaining * 0.4, 30.0)
        
        found = None
        # Multiple restarts with different seeds
        num_restarts = 3
        for restart in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
            budget = base_budget / num_restarts
            seed_adj = make_finite_field_seed(n, 17, 2)
            found = try_find_graph(n, budget, seed_adj=seed_adj)
            if found is not None:
                break

        if found is not None:
            best_n = n
            best_edges = found
            break

    return {'n': best_n, 'edges': best_edges}
