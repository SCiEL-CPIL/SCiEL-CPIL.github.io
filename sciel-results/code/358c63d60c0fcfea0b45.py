import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 110  # Safety margin before 120s limit
    
    best_n = 17
    best_edges = None
    
    # Seed: Paley graph on 17 vertices
    squares = {x * x % 17 for x in range(1, 17)}
    edges_17 = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    best_edges = edges_17
    
    # Helper to check validity exactly
    def is_valid(n, edges):
        # Build adjacency matrix as list of bitmasks
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Check for K4
        # A K4 is a set of 4 vertices where all pairs are connected.
        # Iterate over triangles (u, v, w) with u < v < w.
        # If u,v,w form a triangle, check if there is a vertex x 
        # connected to u, v, and w.
        for u in range(n):
            # Only consider v > u to avoid duplicates
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    continue
                for w in range(v + 1, n):
                    if not (adj[u] >> w) & 1 or not (adj[v] >> w) & 1:
                        continue
                    # u, v, w form a triangle
                    # Common neighbors of u and w
                    common_uw = adj[u] & adj[w]
                    # x must be in common_uw and connected to v
                    # x must be distinct from u, v, w
                    # x is a vertex index
                    candidates = common_uw & adj[v]
                    # Mask out u, v, w
                    candidates &= ~(1 << u)
                    candidates &= ~(1 << v)
                    candidates &= ~(1 << w)
                    if candidates:
                        return False
        
        # Check for Independent Set of size 6
        # An independent set is a set of vertices with no edges between them.
        # We can use a backtracking search to find if an independent set of size 6 exists.
        # To speed up, we can use a greedy heuristic first, but for exactness we need backtracking.
        # However, for n <= 35, checking all 6-subsets is feasible if optimized.
        # Number of 6-subsets of 35 is ~1.6M.
        # Let's use a recursive backtracking with pruning.
        
        def find_independent_set(vertices_mask, size):
            if size == 0:
                return True
            if vertices_mask == 0:
                return False
            # If number of bits in mask < size, return False
            if bin(vertices_mask).count('1') < size:
                return False
            
            # Pick a vertex (e.g., the lowest set bit)
            v = (vertices_mask & -vertices_mask).bit_length() - 1
            # Branch 1: Include v
            # Remove v and its neighbors
            new_mask = vertices_mask & ~(1 << v)
            new_mask &= ~adj[v]
            if find_independent_set(new_mask, size - 1):
                return True
            
            # Branch 2: Exclude v
            new_mask2 = vertices_mask & ~(1 << v)
            if find_independent_set(new_mask2, size):
                return True
            
            return False

        all_vertices = (1 << n) - 1
        if find_independent_set(all_vertices, 6):
            return False
            
        return True

    # Try to find a graph with n > 17
    for n in range(18, 36):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for this n
        found = False
        attempts = 0
        max_attempts = 100 # Adjust based on time
        
        while not found and time.time() < deadline:
            attempts += 1
            # Random initialization
            # Probability of edge p. 
            # For K4-free, we want sparse-ish? 
            # For no independent 6, we want dense-ish?
            # Turan graph T(35, 3) is K4-free. 
            # Independence number of T(n, 3) is n/3 ~ 11. Too high.
            # We need a graph that is "random-like" but structured.
            
            # Let's try a random graph with p=0.5
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        edges.append([i, j])
                        
            # Local Search: Flip edges to reduce violations
            # We don't have a fast violation counter, so we just check validity.
            # If invalid, we can try to fix specific violations.
            # But finding violations is as hard as checking validity.
            
            # Alternative: Use a heuristic construction.
            # Start with a Turan graph and add edges? No, Turan is K4 free but has large indep sets.
            # Start with a random graph and remove edges to break K4s?
            
            # Let's just check the random graph.
            if is_valid(n, edges):
                best_n = n
                best_edges = edges
                found = True
                break
            
            # If not found, try a different strategy: 
            # Take a known valid graph on n-1 and add a vertex.
            # This is complex to implement correctly in the loop.
            
            # Let's try a greedy construction for the next attempt
            # Start with n-1 valid graph? We don't have it stored for all n.
            
            # Let's try a specific structure: Circulant graph
            # For prime n, try quadratic residues if n % 4 == 1
            if n % 4 == 1 and n < 40:
                squares = {x * x % n for x in range(1, n)}
                circ_edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n in squares or (i - j) % n in squares]
                if is_valid(n, circ_edges):
                    best_n = n
                    best_edges = circ_edges
                    found = True
                    break

            # If time is running low, break
            if time.time() > deadline - 10:
                break

    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
