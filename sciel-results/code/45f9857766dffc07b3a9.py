"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np

N = 64

# EVOLVE-BLOCK-START
def construct():
    # Generate grid points
    points = [(x, y) for x in range(N) for y in range(N)]
    num_points = len(points)
    
    # Precompute squared distances and neighbors
    # neighbors[i][d2] = list of indices j such that dist(points[i], points[j])^2 = d2
    neighbors = [{} for _ in range(num_points)]
    
    for i in range(num_points):
        x1, y1 = points[i]
        for j in range(num_points):
            x2, y2 = points[j]
            dx = x1 - x2
            dy = y1 - y2
            d2 = dx*dx + dy*dy
            if d2 not in neighbors[i]:
                neighbors[i][d2] = []
            neighbors[i][d2].append(j)
            
    # Build Conflict Graph as bitsets
    # adj[i] is an integer where bit j is 1 if i and j conflict
    adj = [0] * num_points
    
    # Helper to add edge
    def add_edge(u, v):
        if u == v: return
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Type 1: w is apex. dist(u, w) == dist(v, w)
    # For each w, for each distance d2, all pairs in neighbors[w][d2] conflict.
    for w in range(num_points):
        for d2, lst in neighbors[w].items():
            k = len(lst)
            if k < 2: continue
            # Add edges between all pairs in lst
            for a in range(k):
                u = lst[a]
                for b in range(a+1, k):
                    v = lst[b]
                    add_edge(u, v)

    # Type 2: u is apex. dist(u, w) == dist(u, v)
    # For each u, for each w, let d2 = dist(u,w)^2.
    # All v in neighbors[u][d2] conflict with w.
    for u in range(num_points):
        for w in range(num_points):
            if w == u: continue
            x1, y1 = points[u]
            x2, y2 = points[w]
            dx = x1 - x2
            dy = y1 - y2
            d2 = dx*dx + dy*dy
            
            # v must be in neighbors[u][d2]
            if d2 in neighbors[u]:
                for v in neighbors[u][d2]:
                    if v == w: continue
                    add_edge(w, v)
                    # Note: This covers the case where u is the apex.
                    # The triangle is (w, v, u). Sides uw and uv are equal.
                    # So w and v are the base. They conflict.

    # Solve Maximum Independent Set (MIS) using Greedy Heuristic with Restarts
    def greedy_mis(seed):
        random.seed(seed)
        remaining = set(range(num_points))
        selected = []
        
        while remaining:
            # Pick node with minimum degree in remaining graph
            # To speed up, we can just pick randomly from a subset or use a heuristic
            # For small N, min-degree is fine.
            min_deg_node = -1
            min_deg = float('inf')
            
            # Optimization: If we just pick random, it might be faster but worse.
            # Let's try min-degree.
            for u in remaining:
                # Count neighbors in remaining
                # adj[u] is a big int. We need to mask it with remaining bits.
                # Creating a mask for 'remaining' is O(N).
                # Instead, we can just iterate and check if bit is set.
                deg = 0
                # This is O(N) per node, so O(N^2) per step. N=4096. 16M ops per step.
                # With ~100 steps, 1.6B ops. Might be slow in Python.
                
                # Faster: Precompute a mask?
                # Let's use a simple heuristic: Pick random node.
                # Or pick node with fewest total neighbors (static degree).
                # Static degree is fixed.
                pass
            
            # Better Greedy: Pick node with lowest static degree among remaining.
            # Precompute static degrees
            static_deg = [bin(adj[i]).count('1') for i in range(num_points)]
            
            # Sort remaining by static degree?
            # Let's just pick the one with min static degree in remaining.
            candidates = list(remaining)
            random.shuffle(candidates) # Randomize to break ties
            
            best_u = -1
            best_deg = float('inf')
            for u in candidates:
                if static_deg[u] < best_deg:
                    best_deg = static_deg[u]
                    best_u = u
            
            selected.append(best_u)
            
            # Remove best_u and its neighbors from remaining
            to_remove = set([best_u])
            # Get neighbors from bitset
            mask = adj[best_u]
            # Iterate through bits set in mask
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if idx in remaining:
                    to_remove.add(idx)
                m ^= lsb
            
            remaining -= to_remove
            
        return selected

    best_solution = []
    # Run multiple times with different seeds
    for seed in range(10):
        sol = greedy_mis(seed)
        if len(sol) > len(best_solution):
            best_solution = sol
            
    # Return the points
    return [points[i] for i in best_solution]

# EVOLVE-BLOCK-END
