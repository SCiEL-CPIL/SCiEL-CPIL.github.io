import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Target n = 35
    # If 35 fails, try 34, 33...
    
    def get_adjacency_bitmasks(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def edges_from_bitmasks(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            # Only consider j > i to avoid duplicates
            m = mask & ~((1 << (i + 1)) - 1)
            while m:
                # Get lowest set bit
                lsb = m & -m
                j = lsb.bit_length() - 1
                edges.append([i, j])
                m ^= lsb
        return edges

    def has_k4(adj, n):
        """Returns (i, j, k, l) if K4 found, else None."""
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    # Common neighbors of i and j, greater than j
                    common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
                    if common:
                        # Check if any pair in common is connected
                        # Iterate through bits in common
                        m = common
                        first_k = -1
                        first_k_bit = 0
                        while m:
                            lsb = m & -m
                            k = lsb.bit_length() - 1
                            if first_k == -1:
                                first_k = k
                                first_k_bit = lsb
                            else:
                                # Check if k and first_k are connected
                                if adj[k] & first_k_bit:
                                    return (i, j, first_k, k)
                            m ^= lsb
        return None

    def has_independent_set_6(adj, n):
        """Returns a tuple of 6 vertices forming an independent set, else None."""
        # Backtracking search
        # candidates is a bitmask of available vertices
        # current is a list of vertices in the set
        
        def backtrack(candidates, current):
            if len(current) == 6:
                return tuple(current)
            
            # Pruning: if not enough candidates left
            if candidates.bit_count() + len(current) < 6:
                return None
            
            # Heuristic: pick a vertex with high degree in the induced subgraph of candidates?
            # Or just pick the lowest index for determinism?
            # Randomization might help escape local minima, but for finding *one* set,
            # structured search is often better.
            # Let's pick a vertex v from candidates.
            # To speed up, we can try to find a vertex that is "easy" to include.
            # Vertices with few neighbors in candidates are good candidates for inclusion.
            
            # Let's just iterate through candidates
            m = candidates
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                
                # Option 1: Include v
                # New candidates: candidates without v and without neighbors of v
                new_candidates = candidates & ~(adj[v] | lsb)
                res = backtrack(new_candidates, current + [v])
                if res:
                    return res
                
                # Option 2: Exclude v
                # This branch is usually less promising for finding a large set quickly
                # unless we are doing a full search.
                # For finding *one* set, "Include" first is usually better if we order by "includability".
                # However, if we always include, we might miss a set if the first choice is bad.
                # To keep it fast, we'll rely on the fact that if alpha >= 6, a greedy-ish approach often finds it.
                # If we want to be robust, we should explore both.
                # But exploring "Exclude" is expensive.
                # Let's try a randomized approach: shuffle candidates?
                # Or just try "Include" first. If it fails, try "Exclude".
                
                res = backtrack(candidates & ~lsb, current)
                if res:
                    return res
                
                m ^= lsb
            return None
        
        return backtrack((1 << n) - 1, [])

    def is_valid(adj, n):
        if has_k4(adj, n):
            return False
        if has_independent_set_6(adj, n):
            return False
        return True

    def random_graph(n, p):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    edges.append([i, j])
        return edges

    def local_search(n, initial_edges=None, max_iters=5000):
        if initial_edges is None:
            # Start with a random graph. p=0.6 is a good guess for R(4,6)
            initial_edges = random_graph(n, 0.6)
        
        adj = get_adjacency_bitmasks(n, initial_edges)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Check for violations
            k4 = has_k4(adj, n)
            indep6 = has_independent_set_6(adj, n)
            
            if not k4 and not indep6:
                return adj, edges_from_bitmasks(adj, n)
            
            # If both exist, pick one randomly
            if k4 and indep6:
                if random.random() < 0.5:
                    # Remove edge from K4
                    i, j, k, l = k4
                    # Pick a random edge from the K4
                    edges_in_k4 = [(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)]
                    u, v = random.choice(edges_in_k4)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    # Add edge to Indep Set
                    # Pick a random pair in the independent set
                    verts = indep6
                    u, v = random.sample(verts, 2)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            elif k4:
                i, j, k, l = k4
                edges_in_k4 = [(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)]
                u, v = random.choice(edges_in_k4)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            elif indep6:
                verts = indep6
                u, v = random.sample(verts, 2)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            # Occasionally perturb if stuck?
            # For now, simple hill climbing.
            
        return None, None

    # Try to find a valid graph for n = 35, 34, ...
    for n in range(35, 20, -1):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple random restarts
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Use a slightly different p for diversity
            p = 0.5 + random.random() * 0.2
            
            # Initial random graph
            edges = random_graph(n, p)
            adj = get_adjacency_bitmasks(n, edges)
            
            # Run local search
            adj_res, edges_res = local_search(n, edges, max_iters=2000)
            
            if adj_res is not None:
                # Verify
                if is_valid(adj_res, n):
                    return {'n': n, 'edges': edges_res}
        
    # Fallback: Return a small valid graph if none found
    # Paley graph on 17 vertices is K4-free and alpha < 6?
    # Paley(17): q=17. Squares: 1,4,9,16,8,2,15,13.
    # It is known that Paley(q) is K4-free for q=17?
    # Actually, we can just return a small known valid graph.
    # n=5, empty graph. No K4, no Indep 6.
    return {'n': 5, 'edges': []}

