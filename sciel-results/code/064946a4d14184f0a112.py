import time
import random

def construct():
    start_time = time.time()
    time_limit = 120.0

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        # Initialize adjacency masks
        # adj_mask[i] has bit j set if (i,j) is an edge
        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # We need to track remaining edges to know when to stop and to remove them.
        # But we can also just check if adj_mask is all 0s?
        # No, because we remove edges, adj_mask changes.
        # We can track the number of remaining edges.
        # Or just loop while any adj_mask[i] != 0.
        
        best_cliques = None
        best_count = float('inf')
        
        # Run multiple trials
        # Estimate time per trial
        # n=64, bitmask ops are very fast.
        
        while time.time() - start_time < time_limit - 5.0:
            # Reset
            curr_adj = adj_mask[:]
            remaining_edges_count = len(edges)
            cliques = []
            
            # We need to remove edges from curr_adj
            # And track which edges are removed?
            # Actually, if we just modify curr_adj, we can check if it's empty.
            # But we need to ensure we don't pick the same edge twice.
            # The greedy process picks a clique, removes its edges.
            # The next clique is picked from the remaining graph.
            
            while True:
                # Check if any edges remain
                # Sum of popcounts / 2
                # Or just check if any curr_adj[i] != 0
                active = False
                for i in range(n):
                    if curr_adj[i]:
                        active = True
                        break
                if not active:
                    break
                
                # Pick vertex v with max degree
                max_deg = -1
                candidates = []
                for i in range(n):
                    deg = bin(curr_adj[i]).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        candidates = [i]
                    elif deg == max_deg:
                        candidates.append(i)
                
                if not candidates:
                    break
                    
                v = random.choice(candidates)
                
                # Find maximal clique containing v
                C_mask = 1 << v
                candidates_mask = curr_adj[v]
                
                while candidates_mask:
                    # Pick u in candidates_mask with max degree in curr_adj
                    # Iterate through bits in candidates_mask
                    best_u = -1
                    best_deg = -1
                    mask = candidates_mask
                    while mask:
                        # Get lowest set bit
                        lsb = mask & -mask
                        u = lsb.bit_length() - 1
                        deg = bin(curr_adj[u]).count('1')
                        if deg > best_deg:
                            best_deg = deg
                            best_u = u
                        mask ^= lsb
                    
                    if best_u == -1:
                        break
                    
                    # Add best_u to C
                    C_mask |= (1 << best_u)
                    
                    # Update candidates: intersection with neighbors of best_u
                    candidates_mask &= curr_adj[best_u]
                
                # Extract vertices from C_mask
                C_vertices = []
                mask = C_mask
                while mask:
                    lsb = mask & -mask
                    v_idx = lsb.bit_length() - 1
                    C_vertices.append(v_idx)
                    mask ^= lsb
                
                # Remove edges within C_vertices
                # For each pair in C_vertices, clear bits in curr_adj
                for i in range(len(C_vertices)):
                    u = C_vertices[i]
                    for j in range(i + 1, len(C_vertices)):
                        v = C_vertices[j]
                        # Clear v from u and u from v
                        curr_adj[u] &= ~(1 << v)
                        curr_adj[v] &= ~(1 << u)
                
                cliques.append(C_vertices)
            
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = cliques[:]
        
        return best_cliques if best_cliques else []

    res1 = solve_instance(32, 0.6, 601)
    res2 = solve_instance(64, 0.55, 602)
    
    return [res1, res2]
