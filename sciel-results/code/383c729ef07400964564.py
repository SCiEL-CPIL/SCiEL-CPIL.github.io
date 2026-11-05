import time
import random

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    def check_triangle_free(adj, n):
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    if adj[i] & adj[j]:
                        return False, (i, j)
        return True, None

    def max_independent_set_size(adj, n):
        # Returns the size of the maximum independent set
        # Uses a simple branch and bound with bitmasks
        # This might be slow if not optimized, but n=40 is small enough with pruning
        
        best = 0
        
        def dfs(candidates, current_size):
            nonlocal best
            if not candidates:
                if current_size > best:
                    best = current_size
                return
            
            # Bound: if current_size + popcount(candidates) <= best, prune
            if current_size + bin(candidates).count('1') <= best:
                return
                
            # Pick a vertex
            # Heuristic: pick vertex with max degree in the induced subgraph?
            # For simplicity, pick the lowest set bit
            v = (candidates & -candidates).bit_length() - 1
            mask_v = 1 << v
            
            # Branch 1: Include v
            # Neighbors of v in candidates are removed
            neighbors = adj[v] & candidates
            dfs(candidates & ~mask_v & ~neighbors, current_size + 1)
            
            # Branch 2: Exclude v
            dfs(candidates & ~mask_v, current_size)
            
        dfs((1 << n) - 1, 0)
        return best

    def is_valid(adj, n):
        tf, _ = check_triangle_free(adj, n)
        if not tf:
            return False
        alpha = max_independent_set_size(adj, n)
        return alpha < 10

    def find_graph(n):
        # Local search
        while time.time() < deadline:
            # Initialize random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Check initial
            if is_valid(adj, n):
                return adj
            
            # Local search loop
            for _ in range(1000): # Max steps per restart
                if time.time() > deadline:
                    return None
                
                # Check validity
                if is_valid(adj, n):
                    return adj
                
                # Find a triangle
                tri_found = False
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            common = adj[i] & adj[j]
                            if common:
                                # Found triangle (i, j, k) where k is in common
                                k = (common & -common).bit_length() - 1
                                # Remove random edge from triangle
                                edge_to_remove = random.choice([(i, j), (i, k), (j, k)])
                                u, v = edge_to_remove
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                                tri_found = True
                                break
                    if tri_found:
                        break
                
                if not tri_found:
                    # No triangles, but invalid means alpha >= 10
                    # Find an independent set of size 10
                    # This is expensive, but we can try to find one greedily or via search
                    # For now, if no triangle, we are close.
                    # Let's try to add an edge to break independence
                    # We need to find an independent set S of size 10
                    # We can use a heuristic to find a large independent set
                    # If we find one, add a random edge within it that doesn't create a triangle
                    
                    # Heuristic to find independent set
                    S = []
                    remaining = (1 << n) - 1
                    while remaining and len(S) < 10:
                        # Pick a vertex
                        v = (remaining & -remaining).bit_length() - 1
                        S.append(v)
                        remaining &= ~adj[v]
                        remaining &= ~(1 << v)
                    
                    if len(S) == 10:
                        # Try to add an edge in S
                        added = False
                        for u in S:
                            for v in S:
                                if u < v and not (adj[u] & (1 << v)):
                                    # Check if adding (u,v) creates a triangle
                                    # Common neighbors of u and v
                                    if not (adj[u] & adj[v]):
                                        adj[u] |= (1 << v)
                                        adj[v] |= (1 << u)
                                        added = True
                                        break
                            if added:
                                break
                        if not added:
                            # Could not add edge without triangle, restart or remove edge elsewhere?
                            # For simplicity, restart
                            break
                    else:
                        # Greedy didn't find 10, but maybe it exists?
                        # If exact check says invalid, it exists.
                        # Let's just restart to avoid complex search here
                        break

    # Try n=40 first
    adj_40 = find_graph(40)
    if adj_40 is not None:
        return {'n': 40, 'edges': [[i, j] for i in range(40) for j in range(i + 1, 40) if adj_40[i] & (1 << j)]}
        
    # Try n=39
    adj_39 = find_graph(39)
    if adj_39 is not None:
        return {'n': 39, 'edges': [[i, j] for i in range(39) for j in range(i + 1, 39) if adj_39[i] & (1 << j)]}
        
    # Fallback to known n=26 if all else fails (should not happen)
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
