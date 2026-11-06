import time
import random
import math

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin

    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            m = adj[i]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                m -= lsb
        return edges

    def count_edges(adj):
        return sum(b.bit_count() for b in adj) // 2

    def is_add_valid(u, v, adj):
        """
        Check if adding edge (u,v) keeps the graph K3,3-free.
        When adding (u,v), new K3,3 violations can only involve u or v.
        """
        mask_u = adj[u]
        mask_v = adj[v]
        
        # Check triples {u, i, j} where i,j in N(v)
        # After adding (u,v), N(u) gains v. So for any pair (i,j) in N(v),
        # common neighbors of {u,i,j} = (N(u) ∩ N(i) ∩ N(j)) ∪ {v}
        # We need |N(u) ∩ N(i) ∩ N(j) \ {v}| <= 1
        m = mask_v
        while m:
            lsb_i = m & -m
            i = lsb_i.bit_length() - 1
            m -= lsb_i
            mask_i = adj[i]
            m2 = m  # remaining neighbors of v (already excluded i)
            while m2:
                lsb_j = m2 & -m2
                j = lsb_j.bit_length() - 1
                m2 -= lsb_j
                mask_j = adj[j]
                common = mask_u & mask_i & mask_j
                common &= ~(1 << v)
                if common.bit_count() >= 2:
                    return False

        # Check triples {v, i, j} where i,j in N(u)
        m = mask_u
        while m:
            lsb_i = m & -m
            i = lsb_i.bit_length() - 1
            m -= lsb_i
            mask_i = adj[i]
            m2 = m
            while m2:
                lsb_j = m2 & -m2
                j = lsb_j.bit_length() - 1
                m2 -= lsb_j
                mask_j = adj[j]
                common = mask_v & mask_i & mask_j
                common &= ~(1 << u)
                if common.bit_count() >= 2:
                    return False

        return True

    def is_valid_graph(adj):
        """
        Full check for K3,3.
        """
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                common_ij = ai & adj[j]
                if common_ij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    common_ijk = common_ij & adj[k]
                    if common_ijk.bit_count() >= 3:
                        return False
        return True

    def greedy_add(adj):
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        if is_add_valid(u, v, adj):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            changed = True
        return adj

    def local_search_sa(adj_init):
        """
        Simulated Annealing starting from a valid graph.
        Optimized: precompute edge list, faster move generation.
        """
        adj = adj_init[:]
        current_edges = count_edges(adj)
        best_adj = adj[:]
        best_edges = current_edges

        T = 1.5
        cooling = 0.9993
        min_T = 0.03
        steps_since_improve = 0
        max_steps_no_improve = 3000

        # Precompute list of all possible edges for quick random selection
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        while time.time() < deadline and T > min_T:
            # Randomly add or remove
            if random.random() < 0.5:
                # Try to add
                u, v = random.choice(all_possible)
                if adj[u] & (1 << v):
                    continue
                if is_add_valid(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                        steps_since_improve = 0
                    else:
                        steps_since_improve += 1
            else:
                # Try to remove a random existing edge
                # Build edge list quickly
                edge_list = []
                for i in range(n):
                    m = adj[i]
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        if i < j:
                            edge_list.append((i, j))
                        m -= lsb
                if not edge_list:
                    continue
                u, v = random.choice(edge_list)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_edges -= 1
                steps_since_improve += 1

            T *= cooling

            # Periodic greedy fill
            if random.random() < 0.08 or steps_since_improve > max_steps_no_improve:
                greedy_add(adj)
                new_edges = count_edges(adj)
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj = adj[:]
                steps_since_improve = 0
            
            if steps_since_improve > 15000:
                break

        return best_adj, best_edges

    def construct_deterministic():
        """
        Construct a strong deterministic starting point using a structured approach.
        Based on the idea of using a 5x4 grid-like structure.
        """
        adj = [0] * n
        # Create 4 groups of 5 vertices each
        # Group g: vertices 5*g to 5*g+4
        # Connect vertices in different groups with a pattern
        # Use a 5x4 incidence matrix (each group of 5 connects to 4 groups)
        
        # Pattern: for group i, connect to groups (i+1)%4, (i+2)%4, (i+3)%4, (i+4)%4
        # But we need to avoid K3,3. Let's use a simpler pattern.
        
        # Start with a complete 4-partite graph with parts of size 5
        # But that might have K3,3. Let's be more careful.
        
        # Use a bipartite-like structure between pairs of groups
        for i in range(4):
            for j in range(i+1, 4):
                # Connect group i and group j
                # Use a 5x5 submatrix with at most 2 common neighbors per pair
                # Simple pattern: connect (i,a) to (j,b) if (a-b) mod 5 is in {0, 1, 2}
                for a in range(5):
                    for b in range(5):
                        if (a - b) % 5 in [0, 1, 2]:
                            u = 5 * i + a
                            v = 5 * j + b
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
        
        # Add some edges within groups if valid
        for i in range(4):
            for a in range(5):
                for b in range(a+1, 5):
                    u = 5 * i + a
                    v = 5 * i + b
                    if is_add_valid(u, v, adj):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        
        return adj

    best_global_edges = 0
    best_global_adj = [0] * n

    # Start with deterministic construction
    adj_det = construct_deterministic()
    if is_valid_graph(adj_det):
        greedy_add(adj_det)
        det_edges = count_edges(adj_det)
        if det_edges > best_global_edges:
            best_global_edges = det_edges
            best_global_adj = adj_det[:]

    # Multiple restarts
    num_restarts = 80
    for r in range(num_restarts):
        if time.time() > deadline:
            break

        # Initialize
        if r == 0:
            adj_init = [0] * n
        elif r % 2 == 0:
            # Use deterministic as base
            adj_init = best_global_adj[:] if best_global_edges > 0 else [0] * n
            # Add some random edges
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj_init[i] & (1 << j)):
                        if random.random() < 0.1:
                            adj_init[i] |= (1 << j)
                            adj_init[j] |= (1 << i)
        else:
            adj_init = [0] * n
            # Vary density
            p = 0.2 + 0.05 * (r % 5)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj_init[i] |= (1 << j)
                        adj_init[j] |= (1 << i)
        
        # Repair
        attempts = 0
        while not is_valid_graph(adj_init) and attempts < 2000:
            all_edges = []
            for i in range(n):
                m = adj_init[i]
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    if i < j:
                        all_edges.append((i, j))
                    m -= lsb
            if not all_edges:
                break
            u, v = random.choice(all_edges)
            adj_init[u] &= ~(1 << v)
            adj_init[v] &= ~(1 << u)
            attempts += 1

        if not is_valid_graph(adj_init):
            continue

        # Local Search
        local_adj, local_edges = local_search_sa(adj_init)
        
        if local_edges > best_global_edges:
            best_global_edges = local_edges
            best_global_adj = local_adj[:]

    # Final greedy pass
    if best_global_edges > 0:
        greedy_add(best_global_adj)
        final_edges = count_edges(best_global_adj)
        if final_edges > best_global_edges:
            best_global_edges = final_edges

    if best_global_edges == 0:
        return []

    return get_edges_from_adj(best_global_adj)

if __name__ == "__main__":
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Found {len(edges)} edges in {t1-t0:.2f}s")
    # Verify
    n = 20
    adj = [0]*n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    
    valid = True
    for i in range(n):
        for j in range(i+1, n):
            common = adj[i] & adj[j]
            if common.bit_count() < 3: continue
            for k in range(j+1, n):
                if (common & adj[k]).bit_count() >= 3:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    print(f"Valid K3,3-free: {valid}")
