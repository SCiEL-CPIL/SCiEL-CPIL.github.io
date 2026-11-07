"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Precompute edge list for faster iteration if needed
    edge_list = [(u, v) for u, v in edges]

    def is_valid(coloring):
        # 1. Proper coloring
        for u, v in edge_list:
            if coloring[u] == coloring[v]:
                return False
        
        # 2. Bicolored forest constraint
        colors_used = set(coloring)
        # If only 1 color, it's valid (no pairs)
        if len(colors_used) <= 1:
            return True
            
        # For each pair of colors, check if induced subgraph is a forest
        # We can optimize by only checking pairs that actually have edges between them?
        # No, a cycle can exist even if no direct edge? No, cycle implies edges.
        # But we need to check connectivity.
        
        # Optimization: Only check pairs (c1, c2) where there is at least one edge
        # between a vertex of color c1 and a vertex of color c2.
        # Actually, if there are no edges, it's a forest (disconnected vertices).
        # So we can skip pairs with no edges.
        
        # Build a map: color -> list of vertices
        color_verts = {}
        for i, c in enumerate(coloring):
            if c not in color_verts:
                color_verts[c] = []
            color_verts[c].append(i)
            
        # Identify active pairs
        active_pairs = set()
        for u, v in edge_list:
            cu, cv = coloring[u], coloring[v]
            if cu != cv:
                pair = (min(cu, cv), max(cu, cv))
                active_pairs.add(pair)
                
        # Check each active pair
        for c1, c2 in active_pairs:
            # Get vertices in this pair
            verts = color_verts.get(c1, []) + color_verts.get(c2, [])
            if len(verts) < 3:
                continue # Can't have a cycle
            
            vert_set = set(verts)
            
            # Union-Find
            parent = {v: v for v in verts}
            rank = {v: 0 for v in verts}
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False # Cycle
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True
            
            # Add edges
            for u, v in edge_list:
                if u in vert_set and v in vert_set:
                    if not union(u, v):
                        return False # Cycle found
        
        return True

    def get_cost(coloring):
        # Minimize number of distinct colors
        return len(set(coloring))

    # Simulated Annealing
    # Start with a valid coloring. Identity is valid.
    current_coloring = list(range(n))
    current_cost = get_cost(current_coloring)
    best_coloring = current_coloring[:]
    best_cost = current_cost
    
    start_time = time.time()
    timeout = 55.0 # Leave 5s margin
    
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    while T > min_T and (time.time() - start_time) < timeout:
        # Pick a random vertex
        v = random.randint(0, n - 1)
        old_color = current_coloring[v]
        
        # Pick a new color. 
        # To reduce colors, we should bias towards colors already in use.
        # To increase diversity (if stuck), we might need new colors, but we want to minimize.
        # So, pick from existing colors mostly.
        existing_colors = list(set(current_coloring))
        
        # With high probability, pick from existing colors
        if random.random() < 0.8:
            if existing_colors:
                new_color = random.choice(existing_colors)
            else:
                new_color = 0
        else:
            # Try a new color or a random one in 0..49
            new_color = random.randint(0, 49)
            
        if new_color == old_color:
            continue
            
        # Create new coloring
        new_coloring = current_coloring[:]
        new_coloring[v] = new_color
        
        # Check validity
        if not is_valid(new_coloring):
            # Reject invalid moves immediately? 
            # Or accept with very low probability? 
            # Since we want to stay in valid space, reject.
            # However, if we reject all invalid, we might get stuck.
            # But moving to an invalid state and back is expensive.
            # Let's try to only move to valid states.
            # If we can't find a valid move, we are stuck.
            # To escape, we might need to allow temporary invalidity? 
            # No, the constraint is hard. 
            # So we must find a sequence of valid moves.
            # If stuck, we can try to "swap" colors or do a 2-opt?
            # For now, just reject invalid.
            continue
            
        new_cost = get_cost(new_coloring)
        delta = new_cost - current_cost
        
        if delta <= 0:
            # Better or equal cost
            current_coloring = new_coloring
            current_cost = new_cost
            if current_cost < best_cost:
                best_coloring = current_coloring[:]
                best_cost = current_cost
        else:
            # Worse cost
            prob = math.exp(-delta / T)
            if random.random() < prob:
                current_coloring = new_coloring
                current_cost = new_cost
                
        T *= cooling_rate
        
    # Return best found
    return [int(c) for c in best_coloring]

# EVOLVE-BLOCK-END
