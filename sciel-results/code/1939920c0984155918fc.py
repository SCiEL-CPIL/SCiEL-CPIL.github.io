import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    # Represent graph as list of bitmasks
    # adj[i] is a bitmask of neighbors of i
    
    def is_triangle_free(adj):
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    # Check if (i, j) has common neighbor
                    if (adj[i] & adj[j]) != 0:
                        return False
        return True

    def is_3_colorable(adj):
        # Backtracking 3-coloring
        colors = [-1] * n
        
        def backtrack(v):
            if v == n:
                return True
            for c in range(3):
                ok = True
                for u in range(v):
                    if adj[v] & (1 << u):
                        if colors[u] == c:
                            ok = False
                            break
                if ok:
                    colors[v] = c
                    if backtrack(v+1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)

    def count_edges(adj):
        count = 0
        for i in range(n):
            count += bin(adj[i]).count('1')
        return count // 2

    # 1. Generate Clebsch Graph
    def get_clebsch():
        vectors = [i for i in range(32) if bin(i).count('1') % 2 == 0]
        adj = [0] * n
        for u_idx in range(n):
            for v_idx in range(u_idx + 1, n):
                xor = vectors[u_idx] ^ vectors[v_idx]
                if bin(xor).count('1') == 4:
                    adj[u_idx] |= (1 << v_idx)
                    adj[v_idx] |= (1 << u_idx)
        return adj

    best_adj = get_clebsch()
    best_count = count_edges(best_adj)
    
    # 2. Local Search / Random Restart
    # Strategy: Start with Clebsch, try to add edges.
    # Since Clebsch is maximal, we need to start elsewhere or do swaps.
    
    # Try random triangle-free graphs
    # To get dense ones, start with K_8,8 and remove edges? No, stays bipartite.
    # Start with Clebsch and do random edge swaps?
    # Or just random generation biased towards density.
    
    for _ in range(10000): # Limit iterations
        if time.time() > deadline:
            break
        
        # Random graph generation
        adj = [0] * n
        # Add edges randomly, ensuring no triangle
        # To make it dense, we might need to be clever.
        # Just random addition is slow to get dense.
        
        # Alternative: Start with Clebsch, remove random edges, add random edges?
        # This is a random walk.
        
        # Let's try a different base:
        # If there is a 4-chromatic graph with 47 edges, it's very dense.
        # It must be close to K_8,8.
        # K_8,8 is bipartite.
        # To make it non-bipartite, we need an odd cycle.
        # But K_8,8 has no odd cycles.
        # So we must modify the structure significantly.
        
        pass # Placeholder for search logic

    return [ [u, v] for u in range(n) for v in range(u+1, n) if best_adj[u] & (1 << v) ]
