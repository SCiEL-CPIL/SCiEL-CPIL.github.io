"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np
import time

N = 64

# EVOLVE-BLOCK-START
def construct():
    # Generate grid points
    points = [(x, y) for x in range(N) for y in range(N)]
    num_points = len(points)
    
    # Map point to index
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # Build Conflict Graph
    # adj[i] is a set of neighbors
    adj = [set() for _ in range(num_points)]
    
    # For each center c, group points by distance
    # Points at the same distance from c form an isosceles triangle with c as apex
    # So all pairs in the group conflict.
    
    # Optimization: Use NumPy for distance calculation
    P = np.array(points)
    
    for c_idx in range(num_points):
        c = P[c_idx]
        # Compute squared distances from c to all points
        dx = P[:, 0] - c[0]
        dy = P[:, 1] - c[1]
        d2 = dx*dx + dy*dy
        
        # We only care about points at distance > 0
        # Group indices by d2
        # d2 is an array of ints
        # We can use a dictionary to group indices
        dist_groups = {}
        for i in range(num_points):
            if i == c_idx:
                continue
            d = d2[i]
            if d not in dist_groups:
                dist_groups[d] = []
            dist_groups[d].append(i)
            
        # For each group, add edges between all pairs
        for d, lst in dist_groups.items():
            k = len(lst)
            if k < 2:
                continue
            # Add edges between all pairs in lst
            for a in range(k):
                u = lst[a]
                for b in range(a+1, k):
                    v = lst[b]
                    adj[u].add(v)
                    adj[v].add(u)
                    
    # Convert adjacency sets to bitmasks for faster operations
    # num_points = 4096, so we need 4096-bit integers
    adj_bits = [0] * num_points
    for i in range(num_points):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_bits[i] = mask
        
    # Greedy MIS with Restarts
    def greedy_mis(seed):
        random.seed(seed)
        # remaining as a bitmask
        remaining = (1 << num_points) - 1
        selected = []
        
        while remaining:
            # Find a node in remaining with minimum degree in remaining
            # Heuristic: Try a few random nodes and pick the one with min degree
            # Or iterate all? M=4096.
            # Let's try to find min degree efficiently.
            
            # Get list of nodes in remaining
            # This is O(M)
            nodes = []
            m = remaining
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                nodes.append(idx)
                m ^= lsb
                
            if not nodes:
                break
                
            # Pick node with min degree
            # Degree of u in remaining = popcount(adj_bits[u] & remaining)
            min_deg_node = -1
            min_deg = float('inf')
            
            # Optimization: Sample if M is large?
            # For M=4096, iterating all is ~4096 ops.
            # Number of steps is |MIS|.
            # If |MIS| is small, this is fine.
            
            for u in nodes:
                # Count bits
                deg = bin(adj_bits[u] & remaining).count('1')
                if deg < min_deg:
                    min_deg = deg
                    min_deg_node = u
                    
            # Select min_deg_node
            selected.append(min_deg_node)
            
            # Remove min_deg_node and its neighbors from remaining
            mask_to_remove = adj_bits[min_deg_node] | (1 << min_deg_node)
            remaining &= ~mask_to_remove
            
        return selected

    best_solution = []
    # Run multiple times with different seeds
    # Time limit check
    start_time = time.time()
    max_time = 250  # Leave 50s for safety
    
    seed_counter = 0
    while time.time() - start_time < max_time:
        sol = greedy_mis(seed_counter)
        if len(sol) > len(best_solution):
            best_solution = sol
        seed_counter += 1
        
    # Return the points
    return [points[i] for i in best_solution]

# EVOLVE-BLOCK-END
