import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 40
    
    def is_triangle_free(adj, u, v):
        # Check if adding (u, v) creates a triangle
        # i.e., no common neighbor w such that adj[u][w] and adj[v][w]
        # adj is a list of sets or bitmasks for speed
        # Using bitmasks for n=40 fits in int
        common = adj[u] & adj[v]
        return common == 0

    def has_independent_set_10(adj):
        # Returns True if an independent set of size 10 exists
        # Uses backtracking
        # adj is list of int bitmasks. ~adj[u] & ((1<<n)-1) is neighbors of u in complement (non-neighbors in G)
        # Actually, independent set in G is clique in complement.
        # Let's work with G directly.
        # Candidate set C. Pick u in C.
        # Neighbors of u in G cannot be in the independent set.
        # So next candidates are C \ (N(u) U {u}).
        
        mask_all = (1 << n) - 1
        
        def backtrack(candidates, size):
            if size == 10:
                return True
            if candidates == 0:
                return False
            
            # Heuristic: pick vertex with minimum degree in induced subgraph on candidates?
            # Or just pick the first one.
            # For speed, pick the vertex with the most neighbors in candidates (to reduce candidates most).
            # deg(v) in candidates = popcount(adj[v] & candidates)
            # We want to minimize remaining candidates: candidates \ (adj[v] | (1<<v))
            # So we want to MAXIMIZE adj[v] & candidates.
            
            best_v = -1
            max_deg = -1
            temp_cand = candidates
            while temp_cand:
                v = (temp_cand & -temp_cand).bit_length() - 1
                temp_cand &= temp_cand - 1
                deg = (adj[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
                
            # Branch on best_v: include it or not?
            # Standard branching for Independent Set:
            # 1. Include best_v: new_candidates = candidates & ~adj[best_v] & ~(1<<best_v)
            # 2. Exclude best_v: new_candidates = candidates & ~(1<<best_v)
            
            # Try include first?
            if backtrack(candidates & ~adj[best_v] & ~(1 << best_v), size + 1):
                return True
            if backtrack(candidates & ~(1 << best_v), size):
                return True
            return False
        
        return backtrack(mask_all, 0)

    # Generate random maximal triangle-free graphs
    for _ in range(10000):
        if time.time() > deadline:
            break
            
        # Initialize empty graph
        adj = [0] * n
        
        # Generate random edge order
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(edges)
        
        # Add edges if no triangle
        for u, v in edges:
            if is_triangle_free(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        # Check if alpha < 10
        if not has_independent_set_10(adj):
            # Construct edge list
            edge_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        edge_list.append([i, j])
            return {'n': n, 'edges': edge_list}
            
    # Fallback: Return the seed graph (n=26) if we didn't find a 40-vertex one
    # This ensures we always return a valid graph, even if suboptimal
    n_seed = 26
    edges_seed = [[i, j] for i in range(n_seed) for j in range(i + 1, n_seed) if (j - i) % n_seed % 3 == 1]
    return {'n': n_seed, 'edges': edges_seed}
