import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s kill

    # Best known seed: Paley graph on 17 vertices
    squares = {x * x % 17 for x in range(1, 17)}
    edges_17 = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    best_n = 17
    best_edges = edges_17

    # Build adjacency matrix for Paley(17)
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

    def count_edges_in_subgraph(mask, n, adj_masks):
        """Count edges in the subgraph induced by vertices in 'mask'."""
        count = 0
        # Iterate over vertices in mask
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            m ^= lsb
            # Common neighbors of u within the mask
            common = adj_masks[u] & mask
            # Count edges from u to common neighbors
            # We only want to count each edge once, so we can just count the degree of u in the subgraph
            # and sum, then divide by 2.
            # Or, to be safe and simple:
            # count += popcount(common)
            # But this counts each edge twice.
            count += bin(common).count('1')
        return count // 2

    def count_k4s(n, adj_masks):
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj_masks[u] & (1 << v):
                    common = adj_masks[u] & adj_masks[v]
                    total += count_edges_in_subgraph(common, n, adj_masks)
        return total

    def count_ind6s(n, adj_masks):
        total = 0
        full_mask = (1 << n) - 1
        # Complement masks (non-neighbors excluding self)
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(adj_masks[u] | (1 << u))
        
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_masks[u] & (1 << v)):
                    # u and v are non-adjacent
                    # Find common non-neighbors
                    common_non = comp_masks[u] & comp_masks[v]
                    # Count independent 4-sets in common_non
                    # This is equivalent to counting 4-cliques in the complement subgraph
                    # Which is equivalent to counting edges in the complement subgraph induced by pairs?
                    # No, it's counting 4-cliques in \overline{G}[common_non].
                    # We can use the same logic:
                    # Iterate over edges in \overline{G}[common_non]?
                    # Let's define a helper to count 4-cliques in a graph defined by masks.
                    # Actually, we can just count the number of independent 4-sets directly.
                    # An independent 4-set in G[common_non] is a set of 4 vertices with no edges between them.
                    # This is hard to count fast.
                    
                    # Alternative: Count 4-cliques in the complement.
                    # Let comp_adj_masks be the adjacency masks of the complement graph.
                    # We don't have them precomputed for the subgraph.
                    
                    # Let's use the formula:
                    # Number of 4-cliques in H = sum over edges (x,y) in H of E(H[N_H(x) \cap N_H(y)])
                    # Here H = \overline{G}[common_non].
                    # This is getting complex.
                    
                    # For now, let's just count the number of independent 4-sets by iterating over all 4-subsets?
                    # Too slow.
                    
                    # Let's stick to the delta method in SA, where we only need to update the cost.
                    # We will compute the full cost only at the beginning and occasionally.
                    # For the delta, we need to count:
                    # A = E(G[N(u) \cap N(v)])
                    # B = Ind4(\overline{N}(u) \cap \overline{N}(v) \setminus \{u,v\})
                    
                    # We will implement a function to count independent 4-sets in a given mask.
                    pass
        return total

    # This approach of full counting is too slow for n=35.
    # We will use the delta-based SA.

    def count_ind4_in_mask(mask, n, adj_masks):
        """Count independent 4-sets in the subgraph induced by 'mask'."""
        # This is equivalent to counting 4-cliques in the complement subgraph.
        # Let's iterate over pairs (w,x) in mask that are non-adjacent in G.
        # If w,x are non-adjacent, they can be part of an independent set.
        # We need 2 more vertices y,z from mask \ {w,x} such that:
        # y,z are non-adjacent to w,x and to each other.
        
        count = 0
        # Iterate w
        m_w = mask
        while m_w:
            lsb_w = m_w & -m_w
            w = lsb_w.bit_length() - 1
            m_w ^= lsb_w
            
            # Iterate x > w
            m_x = mask & ~((1 << (w + 1)) - 1)
            while m_x:
                lsb_x = m_x & -m_x
                x = lsb_x.bit_length() - 1
                m_x ^= lsb_x
                
                # Check if w,x are non-adjacent
                if adj_masks[w] & (1 << x):
                    continue
                
                # Find common non-neighbors of w and x within mask
                # Non-neighbors of w in mask: mask & ~adj_masks[w]
                # Non-neighbors of x in mask: mask & ~adj_masks[x]
                common_non = mask & ~(adj_masks[w] | adj_masks[x] | (1 << w) | (1 << x))
                
                # Now we need to count pairs (y,z) in common_non such that y,z are non-adjacent.
                # This is the number of non-edges in the subgraph induced by common_non.
                # Let k = popcount(common_non).
                # Total pairs = k*(k-1)/2.
                # Edges in subgraph = E(G[common_non]).
                # Non-edges = Total pairs - Edges.
                
                k = bin(common_non).count('1')
                if k < 2:
                    continue
                
                # Count edges in G[common_non]
                e_count = 0
                m_y = common_non
                while m_y:
                    lsb_y = m_y & -m_y
                    y = lsb_y.bit_length() - 1
                    m_y ^= lsb_y
                    e_count += bin(adj_masks[y] & common_non).count('1')
                e_count //= 2
                
                non_edges = (k * (k - 1)) // 2 - e_count
                count += non_edges
                
        return count

    def delta_cost(n, adj_masks, u, v, adding):
        """Calculate change in cost when toggling edge (u,v)."""
        # A = E(G[N(u) \cap N(v)])
        common_neigh = adj_masks[u] & adj_masks[v]
        A = count_edges_in_subgraph(common_neigh, n, adj_masks)
        
        # B = Ind4(\overline{N}(u) \cap \overline{N}(v) \setminus \{u,v\})
        full_mask = (1 << n) - 1
        comp_u = full_mask & ~(adj_masks[u] | (1 << u))
        comp_v = full_mask & ~(adj_masks[v] | (1 << v))
        common_non = comp_u & comp_v
        B = count_ind4_in_mask(common_non, n, adj_masks)
        
        if adding:
            return A - B
        else:
            return B - A

    def is_valid(n, adj_masks):
        # Check for K4
        for u in range(n):
            for v in range(u + 1, n):
                if adj_masks[u] & (1 << v):
                    common = adj_masks[u] & adj_masks[v]
                    if count_edges_in_subgraph(common, n, adj_masks) > 0:
                        return False
        # Check for Ind 6
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(adj_masks[u] | (1 << u))
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_masks[u] & (1 << v)):
                    common_non = comp_masks[u] & comp_masks[v]
                    if count_ind4_in_mask(common_non, n, adj_masks) > 0:
                        return False
        return True

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
        
        # Initial cost
        # We don't need the exact cost for SA, just deltas.
        # But we need to know if we are valid.
        if is_valid(n, masks):
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            return edges

        T = 5.0
        min_T = 0.01
        cooling = 0.999
        no_improve = 0
        max_no_improve = 10000

        while time.time() < end_time and no_improve < max_no_improve:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue

            old_state = adj[u, v]
            adding = (old_state == 0)

            delta = delta_cost(n, masks, u, v, adding)

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

        if is_valid(n, masks):
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            return edges

        return None

    # Strategy: try to find a valid graph for n=35, then 34, etc.
    for n in range(35, 17, -1):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 5.0:
            break

        # Allocate time
        budget = min(remaining * 0.5, 20.0)
        
        found = None
        # Try a few restarts
        for restart in range(3):
            if time.time() > deadline - 2.0:
                break
            seed_adj = None
            if restart == 0 and n >= 17:
                 # Use Paley seed if possible? 
                 # Paley(17) is too small to seed n=35 directly without expansion.
                 # Just use random.
                 pass
            found = try_find_graph(n, budget / 3.0, seed_adj=seed_adj)
            if found is not None:
                break

        if found is not None:
            best_n = n
            best_edges = found
            break

    return {'n': best_n, 'edges': best_edges}
