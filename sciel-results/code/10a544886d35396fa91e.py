import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 20
    # Precompute all possible edges
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_pairs = len(all_pairs)
    
    best_edges = []
    best_score = 0
    
    # Helper to check if adding (u, v) is valid
    # adj[u] is an integer bitmask of neighbors
    def can_add(adj, u, v):
        # Check if u and v share >= 3 common neighbors
        # common = adj[u] & adj[v]
        # But we must ensure we don't count u or v themselves if they are in the mask?
        # In a simple graph, adj[u] does not contain u. 
        # If we add (u,v), u is added to adj[v] and v to adj[u].
        # The common neighbors of u and v in the NEW graph are the common neighbors in the OLD graph, 
        # PLUS potentially each other? No, u is not a neighbor of u. 
        # The condition "no two vertices have three common neighbours" applies to the final graph.
        # If we add (u,v), the common neighbors of u and v are (N(u) \ {v}) \cap (N(v) \ {u}).
        # Since u and v were not connected before, v is not in N(u) and u is not in N(v).
        # So N(u) and N(v) in the old graph do not contain v and u respectively.
        # Thus, common neighbors in new graph = (adj[u] & adj[v]) in old graph.
        common = adj[u] & adj[v]
        return common.bit_count() < 3

    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            # Only consider v > u to avoid duplicates
            # Iterate through bits set in mask
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                if v > u:
                    edges.append([u, v])
                m ^= lsb
        return edges

    def greedy_add_pass(adj, order):
        """Try to add edges in the given order if valid."""
        added_any = False
        for u, v in order:
            if not (adj[u] >> v) & 1:  # If edge not present
                if can_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added_any = True
        return added_any

    def local_search_add(adj):
        """Repeatedly try to add any valid missing edge until no more can be added."""
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            for u, v in all_pairs:
                if not (adj[u] >> v) & 1:
                    if can_add(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        changed = True
                        # We don't break here, we keep checking others in this pass
                        # This helps find multiple non-conflicting additions in one pass

    # Main Loop
    while time.time() < deadline - 10.0:
        # 1. Initialize with random greedy construction
        adj = [0] * n
        random.shuffle(all_pairs)
        greedy_add_pass(adj, all_pairs)
        
        # 2. Local search to maximize edges from this state
        local_search_add(adj)
        
        current_edges = get_edges_from_adj(adj)
        current_score = len(current_edges)
        
        if current_score > best_score:
            best_score = current_score
            best_edges = current_edges
            
        # 3. Perturbation / Simulated Annealing step
        # If we have time, try to escape local optimum by removing a random edge
        # and re-running local search.
        if time.time() < deadline - 20.0 and current_score > 0:
            # Copy state
            adj_pert = adj[:]
            # Remove a random existing edge
            existing_edges = get_edges_from_adj(adj_pert)
            if existing_edges:
                rem_u, rem_v = random.choice(existing_edges)
                adj_pert[rem_u] &= ~(1 << rem_v)
                adj_pert[rem_v] &= ~(1 << rem_u)
                
                # Try to add back or add others
                local_search_add(adj_pert)
                
                pert_edges = get_edges_from_adj(adj_pert)
                pert_score = len(pert_edges)
                
                if pert_score > best_score:
                    best_score = pert_score
                    best_edges = pert_edges

    return best_edges
