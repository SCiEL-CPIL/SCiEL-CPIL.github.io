import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    def check_c4(adj, u, v):
        """
        Check if adding edge (u, v) creates a C4.
        adj is a list of int bitmasks.
        Returns True if safe (no C4 created), False otherwise.
        """
        # Common neighbors of u and v
        common = adj[u] & adj[v]
        # If there are 2 or more common neighbors, a C4 exists
        # (u, w1, v, w2, u)
        # We need to check if the number of common neighbors is >= 2
        # Note: u and v are not neighbors yet, so they are not in each other's N.
        return (common & (common - 1)) != 0 # Check if at least 2 bits are set

    def max_independent_set(adj, n):
        """
        Exact maximum independent set using branch and bound with bitmasks.
        """
        # Precompute degrees or use simple ordering
        # Order vertices by degree (descending) to prune better
        # deg[i] is popcount of adj[i]
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        # Map original index to position in order?
        # Or just iterate in order.
        # We need to track which vertices are available.
        # Let's use a recursive function on the index in the ordered list.
        
        best = 0
        
        def dfs(idx, current_size, available_mask):
            nonlocal best
            
            if current_size > best:
                best = current_size
            
            if idx == n:
                return
            
            # Pruning: if current_size + remaining vertices <= best, return
            # remaining vertices = n - idx
            if current_size + (n - idx) <= best:
                return
            
            v = order[idx]
            v_bit = 1 << v
            
            # If v is not in available_mask, it's already excluded
            if not (available_mask & v_bit):
                dfs(idx + 1, current_size, available_mask)
                return
            
            # Option 1: Include v
            # New available mask: remove v and all its neighbors
            new_mask = available_mask & ~(adj[v] | v_bit)
            dfs(idx + 1, current_size + 1, new_mask)
            
            # Option 2: Exclude v
            new_mask_excl = available_mask & ~v_bit
            dfs(idx + 1, current_size, new_mask_excl)
            
        dfs(0, 0, (1 << n) - 1)
        return best

    def solve_once(seed_val):
        """
        Run one instance of the greedy construction + vertex removal.
        Returns (n, edges) of the best graph found in this run.
        """
        random.seed(seed_val)
        max_n = 42
        best_n = 0
        best_edges = []
        
        # Greedy edge addition
        # We will try to build a graph on max_n vertices
        # Start with empty graph
        adj = [0] * max_n
        
        # Generate all possible edges
        edges_list = [(i, j) for i in range(max_n) for j in range(i + 1, max_n)]
        random.shuffle(edges_list)
        
        for u, v in edges_list:
            if check_c4(adj, u, v):
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Check independence number
        alpha = max_independent_set(adj, max_n)
        
        # If alpha < 11, we have a valid graph on 42 vertices
        if alpha < 11:
            # Extract edges
            edges = []
            for i in range(max_n):
                for j in range(i + 1, max_n):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            return max_n, edges
        
        # If alpha >= 11, we need to remove vertices
        # We will iteratively remove vertices until alpha < 11
        # To be efficient, we can just remove random vertices
        # and re-check alpha.
        # Since n is small, this is fast.
        
        current_n = max_n
        # active vertices mask
        active = (1 << max_n) - 1
        
        # We need to map the active vertices to a contiguous 0..k-1 for the alpha check?
        # Or just pass the subgraph.
        # It's easier to maintain a list of active vertices and build the subgraph.
        
        active_vertices = list(range(max_n))
        
        while current_n > 0:
            # Check time
            if time.time() > deadline - 5:
                break
                
            # Compute alpha on the subgraph induced by active_vertices
            # Build adj_sub
            n_sub = len(active_vertices)
            if n_sub == 0: break
            
            # Map original index to sub index
            idx_map = {v: i for i, v in enumerate(active_vertices)}
            adj_sub = [0] * n_sub
            for i, u in enumerate(active_vertices):
                for j, v in enumerate(active_vertices):
                    if i < j and ((adj[u] >> v) & 1):
                        adj_sub[i] |= (1 << j)
                        adj_sub[j] |= (1 << i)
                        
            alpha_sub = max_independent_set(adj_sub, n_sub)
            
            if alpha_sub < 11:
                # Valid graph
                if n_sub > best_n:
                    best_n = n_sub
                    # Extract edges for this subgraph
                    edges = []
                    for i, u in enumerate(active_vertices):
                        for j, v in enumerate(active_vertices):
                            if i < j and ((adj[u] >> v) & 1):
                                edges.append([u, v]) # Use original indices
                    best_edges = edges
                break
            else:
                # Remove a vertex
                # Heuristic: remove a vertex with high degree in the subgraph?
                # Or random.
                # Let's try removing a random vertex to explore different structures.
                # Or the one with the highest degree in the subgraph to reduce alpha?
                # Removing high degree vertices might reduce alpha faster.
                degs = [bin(adj_sub[i]).count('1') for i in range(n_sub)]
                # Pick max degree
                max_deg_idx = max(range(n_sub), key=lambda x: degs[x])
                # Remove it
                removed_v = active_vertices[max_deg_idx]
                active_vertices.remove(removed_v)
                current_n -= 1
                # We don't need to update adj (original) as we rebuild adj_sub each time.
                # But we can update active mask.
        
        return best_n, best_edges

    # Run multiple instances with different seeds
    best_global_n = 0
    best_global_edges = []
    
    # Use a few seeds
    seeds = [1, 42, 123, 999, 2023]
    
    for s in seeds:
        if time.time() > deadline - 10:
            break
            
        n, edges = solve_once(s)
        if n > best_global_n:
            best_global_n = n
            best_global_edges = edges
            
        # If we found 42, we can stop early? 
        # 42 is the max possible (2 * 21).
        if best_global_n == 42:
            break

    if best_global_n == 0:
        # Fallback: return a small valid graph
        # e.g., 10 vertices, no edges. Alpha=10 < 11. C4-free.
        return {'n': 10, 'edges': []}
        
    return {'n': best_global_n, 'edges': best_global_edges}
