def construct():
    import time
    import random
    import numpy as np
    
    start_time = time.time()
    deadline = start_time + 110
    
    def is_triangle_free(adj_bits, n):
        # adj_bits[i] is an integer representing neighbors of i
        for i in range(n):
            for j in range(i+1, n):
                if (adj_bits[i] >> j) & 1:
                    # Check common neighbors
                    common = adj_bits[i] & adj_bits[j]
                    if common:
                        return False
        return True
        
    def alpha_exact(adj_bits, n):
        # Branch and bound for max independent set
        # Use complement: max clique in complement
        # comp_adj[i] = (~adj_bits[i]) & ((1 << n) - 1)
        # Remove self
        best = 0
        
        def dfs(candidates, current_size):
            nonlocal best
            if current_size + len(candidates) <= best:
                return
            if not candidates:
                best = max(best, current_size)
                return
                
            # Pick vertex with max degree in induced subgraph?
            # Or just first
            v = candidates[0]
            rest = candidates[1:]
            
            # Include v
            # Neighbors of v in candidates
            # In independent set, we can only pick non-neighbors?
            # No, in G, independent set means no edges.
            # So if we pick v, we can only pick non-neighbors of v from candidates.
            # non_neighbors = candidates & ~adj_bits[v]
            new_cands = [u for u in rest if not ((adj_bits[v] >> u) & 1)]
            dfs(new_cands, current_size + 1)
            
            # Exclude v
            dfs(rest, current_size)
            
            if best >= 10:
                return

        dfs(list(range(n)), 0)
        return best

    def find_graph(n):
        # Random restarts
        for restart in range(50):
            if time.time() > deadline:
                return None
            
            # Initialize with random triangle-free graph
            # Start with empty, add random edges if triangle-free
            adj_bits = [0] * n
            for _ in range(n * n // 2):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if u > v: u, v = v, u
                if (adj_bits[u] >> v) & 1: continue
                # Check triangle
                # If add u-v, check if u and v have common neighbor
                common = adj_bits[u] & adj_bits[v]
                if common:
                    continue
                # Add edge
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                
            # Local search
            for step in range(1000):
                if time.time() > deadline:
                    return None
                
                alpha = alpha_exact(adj_bits, n)
                if alpha < 10:
                    # Convert to edge list
                    edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if (adj_bits[i] >> j) & 1:
                                edges.append([i, j])
                    return edges
                    
                # Try to flip an edge
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if u > v: u, v = v, u
                
                if (adj_bits[u] >> v) & 1:
                    # Remove edge
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
                else:
                    # Add edge if triangle-free
                    common = adj_bits[u] & adj_bits[v]
                    if common:
                        continue
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
        return None

    for n in range(39, 25, -1):
        if time.time() > deadline:
            break
        edges = find_graph(n)
        if edges is not None:
            return {'n': n, 'edges': edges}
            
    # Fallback
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
