import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 115.0  # Keep 5s buffer for 120s limit
    
    # Target n
    # Best known is 39.
    # We will try to construct a graph on n=39.
    # If that fails, try n=38, etc.
    
    def has_c4(adj, u, v):
        # Check if adding edge (u, v) creates a C4.
        # This happens if N(u) and N(v) have a common neighbor.
        # adj is a list of sets (or bitsets)
        # For speed, use bitsets if n is small.
        # n <= 43, so int bitsets are perfect.
        pass

    # Let's use bitsets for speed.
    # adj[i] is an integer where bit j is set if (i,j) is an edge.
    
    def try_construct(n):
        # Initialize
        adj = [0] * n
        # We will add edges greedily.
        # To make it random, we shuffle the order of edge additions.
        # Or, we iterate through vertices and add edges to previous ones.
        
        # Strategy:
        # 1. Create a list of all potential edges.
        # 2. Shuffle them.
        # 3. Iterate through edges.
        # 4. If adding the edge doesn't create a C4, add it.
        # 5. After all edges are considered, check alpha.
        
        # But this might result in a sparse graph.
        # Better:
        # 1. Start with empty graph.
        # 2. While there are edges that can be added without C4:
        #    - Pick a random valid edge.
        #    - Add it.
        
        # C4 check:
        # Edge (u, v) is valid if (adj[u] & adj[v]) == 0.
        
        # We need to find if alpha < 11.
        # We can use a simple backtracking to find an independent set of size 11.
        # If found, we "repair" by adding an edge within that set if possible.
        
        # Let's implement a "Repair" loop.
        
        # 1. Greedily add random valid edges until maximal.
        # 2. Check alpha.
        # 3. If alpha >= 11, find an independent set I.
        # 4. Try to add an edge within I.
        # 5. If successful, go to 2 (or just continue adding).
        # 6. If no edge in I can be added, we are stuck. Return False.
        
        # To speed up, we can limit the number of restarts.
        
        def find_independent_set(adj, k):
            # Returns a list of k vertices that are independent, or None.
            # Uses backtracking.
            # For n=39, k=11, this should be fast if the graph is dense enough.
            # If the graph is sparse, it might be slow, but we are adding edges to make it dense.
            
            # Optimization: Use bitsets.
            # remaining = all vertices.
            # We want to pick a vertex v, remove v and its neighbors.
            
            # Let's use a simple DFS.
            # To make it fast, we can use a heuristic: pick the vertex with the highest degree in the remaining graph?
            # Or just the first one.
            
            def dfs(remaining_mask, current_set):
                if len(current_set) == k:
                    return current_set
                if not remaining_mask:
                    return None
                # Pruning: if number of remaining vertices < k - len(current_set), return None
                if remaining_mask.bit_count() < k - len(current_set):
                    return None
                    
                # Pick a vertex
                # Let's pick a random vertex from remaining to diversify search
                # Or pick the one with highest degree?
                # For finding *any* independent set, picking a low-degree vertex might be better?
                # Actually, for *breaking* independent sets, we just need to find one.
                
                # Let's pick the lowest set bit for determinism in this sub-step, 
                # but we will restart with different edge orders anyway.
                v = (remaining_mask & -remaining_mask).bit_length() - 1
                # Wait, bit_length of 1 is 1, index 0.
                # If mask is 10 (2), bit_length is 2, index 1. Correct.
                
                # Option 1: Include v
                # Remove v and its neighbors
                new_mask = remaining_mask & ~(1 << v) & ~adj[v]
                res = dfs(new_mask, current_set + [v])
                if res:
                    return res
                
                # Option 2: Exclude v
                new_mask = remaining_mask & ~(1 << v)
                res = dfs(new_mask, current_set)
                if res:
                    return res
                
                return None
                
            return dfs((1 << n) - 1, [])

        # Main loop for this n
        for _ in range(10): # 10 restarts per n
            if time.time() > deadline:
                return None
                
            adj = [0] * n
            # Add edges
            # We will try to add edges in a random order.
            # To ensure we get a "maximal" graph, we keep adding until no more can be added.
            
            # List of all edges
            edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(edges)
            
            for u, v in edges:
                if time.time() > deadline:
                    return None
                # Check if (u, v) can be added
                # Common neighbors of u and v
                if (adj[u] & adj[v]) == 0:
                    # Add edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Now check alpha
            # We want alpha < 11.
            # Find independent set of size 11.
            
            # We might need to repair.
            # If we find an independent set I of size 11, we try to add an edge within I.
            # If we can't, we fail this restart.
            
            # But wait, the greedy addition might have left "holes".
            # Let's just check if alpha < 11.
            # If alpha >= 11, we try to fix it.
            
            # Loop of repair
            repaired = True
            while repaired and time.time() < deadline:
                repaired = False
                ind_set = find_independent_set(adj, 11)
                if not ind_set:
                    # Alpha < 11, success!
                    # Convert to edge list
                    edge_list = []
                    for i in range(n):
                        mask = adj[i]
                        while mask:
                            lsb = mask & -mask
                            j = lsb.bit_length() - 1
                            if i < j:
                                edge_list.append([i, j])
                            mask ^= lsb
                    return {'n': n, 'edges': edge_list}
                
                # We found an independent set of size 11.
                # Try to add an edge within it.
                added_edge = False
                # Shuffle the set to try different edges
                ind_list = list(ind_set)
                random.shuffle(ind_list)
                for idx_u in range(len(ind_list)):
                    for idx_v in range(idx_u + 1, len(ind_list)):
                        u = ind_list[idx_u]
                        v = ind_list[idx_v]
                        # Check if we can add (u, v)
                        if (adj[u] & adj[v]) == 0:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            added_edge = True
                            repaired = True
                            break
                    if added_edge:
                        break
                            
                if not added_edge:
                    # Stuck. Cannot break this independent set without creating C4.
                    # Break the repair loop and restart.
                    break
                    
        return None

    # Try n=39 down to 31
    for n in range(39, 30, -1):
        res = try_construct(n)
        if res:
            return res
            
    # Fallback: Return a small valid graph
    # PG(2, 2) is 7 vertices, alpha=2? No, alpha of PG(2,2) polarity graph is 2?
    # Actually, just return a trivial graph if all fails.
    return {'n': 1, 'edges': []}
