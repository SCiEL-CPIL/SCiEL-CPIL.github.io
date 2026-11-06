import numpy as np
import random
import time
import sys

def has_c6(adj, n):
    """
    Returns True if the graph represented by adj (n x n boolean matrix) contains a C6.
    Returns the tuple of vertices of the C6 if found, else None.
    """
    # adj[i][j] is True if edge exists
    # We look for v0-v1-v2-v3-v4-v5-v0
    # Iterate over all pairs (v0, v1) that are connected
    for v0 in range(n):
        for v1 in range(n):
            if v0 == v1 or not adj[v0, v1]:
                continue
            
            # We need to find v2, v3, v4, v5 such that:
            # v1-v2, v2-v3, v3-v4, v4-v5, v5-v0 are edges
            # v2, v3, v4, v5 distinct and not v0, v1
            
            # To avoid duplicates, we can enforce an ordering? 
            # Not strictly necessary if we just want ONE C6.
            # But to be efficient, we can do a DFS from v1 to v0 of length 4.
            
            # DFS state: (current_node, visited_set, path)
            # Start: (v1, {v0, v1}, [v1])
            # Target: reach v0 with path length 5 (v1, v2, v3, v4, v5)
            
            # Actually, simpler:
            # Iterate v2 in neighbors(v1)
            #   Iterate v3 in neighbors(v2)
            #     Iterate v4 in neighbors(v3)
            #       Iterate v5 in neighbors(v4)
            #         Check if adj[v5, v0] is True
            #         Check if v2, v3, v4, v5 are distinct and not v0, v1
            
            # This is O(D^4) per edge. D is avg degree.
            # For n=17, D~4. 4^4 = 256.
            # 136 edges * 256 = 35k operations. Very fast.
            
            for v2 in range(n):
                if v2 in (v0, v1) or not adj[v1, v2]:
                    continue
                for v3 in range(n):
                    if v3 in (v0, v1, v2) or not adj[v2, v3]:
                        continue
                    for v4 in range(n):
                        if v4 in (v0, v1, v2, v3) or not adj[v3, v4]:
                            continue
                        for v5 in range(n):
                            if v5 in (v0, v1, v2, v3, v4):
                                continue
                            if adj[v4, v5] and adj[v5, v0]:
                                return (v0, v1, v2, v3, v4, v5)
    return None

def construct():
    target_n = 17
    random.seed(42)
    
    best_colors = None
    start_time = time.time()
    
    # Try to find a valid coloring for n=17
    # If it takes too long, we might need to fallback, but let's give it 110s.
    
    for attempt in range(500):
        if time.time() - start_time > 110:
            break
            
        # Random initial coloring
        num_edges = target_n * (target_n - 1) // 2
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Function to get adjacency matrices
        def get_adj_matrices(colors, n):
            adj = [np.zeros((n, n), dtype=bool) for _ in range(4)]
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    c = colors[idx]
                    adj[c][i, j] = True
                    adj[c][j, i] = True
                    idx += 1
            return adj
            
        # Local search
        for iteration in range(50000):
            if time.time() - start_time > 115:
                break
                
            adj = get_adj_matrices(colors, target_n)
            
            # Find a C6 violation
            violation_color = -1
            violation_cycle = None
            for c in range(4):
                cycle = has_c6(adj[c], target_n)
                if cycle:
                    violation_color = c
                    violation_cycle = cycle
                    break
            
            if violation_cycle is None:
                # No violations!
                best_colors = colors[:]
                break
            
            # Recolor one edge in the C6
            # The cycle is (v0, v1, v2, v3, v4, v5)
            # Edges are (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0)
            v0, v1, v2, v3, v4, v5 = violation_cycle
            edges = [(v0, v1), (v1, v2), (v2, v3), (v3, v4), (v4, v5), (v5, v0)]
            
            # Pick a random edge from the cycle
            edge = random.choice(edges)
            u, v = edge
            if u > v:
                u, v = v, u
            
            # Find index in colors
            # Index formula for (u, v) with u < v in n vertices:
            # Sum of (n - i - 1) for i=0 to u-1 + (v - u - 1)
            # Or precompute a map
            # Map: (i, j) -> index
            # i * (2*n - i - 1) // 2 + (j - i - 1) ?
            # Let's just use a helper or compute.
            # For n=17:
            # idx = 0
            # for i in range(u):
            #   idx += n - i - 1
            # idx += v - u - 1
            
            idx = 0
            for i in range(u):
                idx += target_n - i - 1
            idx += v - u - 1
            
            # Change color to a random other color
            old_color = colors[idx]
            new_color = random.randint(0, 3)
            while new_color == old_color:
                new_color = random.randint(0, 3)
            colors[idx] = new_color
            
        if best_colors is not None:
            break
            
    if best_colors is not None:
        return {'n': target_n, 'colors': best_colors}
        
    # Fallback: Try n=16
    target_n = 16
    for attempt in range(200):
        if time.time() - start_time > 118:
            break
        num_edges = target_n * (target_n - 1) // 2
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        def get_adj_matrices(colors, n):
            adj = [np.zeros((n, n), dtype=bool) for _ in range(4)]
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    c = colors[idx]
                    adj[c][i, j] = True
                    adj[c][j, i] = True
                    idx += 1
            return adj
            
        for iteration in range(50000):
            if time.time() - start_time > 119:
                break
            adj = get_adj_matrices(colors, target_n)
            violation_cycle = None
            for c in range(4):
                cycle = has_c6(adj[c], target_n)
                if cycle:
                    violation_cycle = cycle
                    break
            if violation_cycle is None:
                best_colors = colors[:]
                break
            
            v0, v1, v2, v3, v4, v5 = violation_cycle
            edges = [(v0, v1), (v1, v2), (v2, v3), (v3, v4), (v4, v5), (v5, v0)]
            edge = random.choice(edges)
            u, v = edge
            if u > v:
                u, v = v, u
            
            idx = 0
            for i in range(u):
                idx += target_n - i - 1
            idx += v - u - 1
            
            old_color = colors[idx]
            new_color = random.randint(0, 3)
            while new_color == old_color:
                new_color = random.randint(0, 3)
            colors[idx] = new_color
            
        if best_colors is not None:
            return {'n': target_n, 'colors': best_colors}

    # Fallback to n=9 (seed level)
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
