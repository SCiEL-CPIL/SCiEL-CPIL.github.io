import time
import random

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_n = 1
    best_colors = []
    
    # Helper to check if a coloring is valid (for final verification)
    def is_valid(n, colors):
        # colors is a list of length n*(n-1)//2
        # Map edge (i, j) to index
        def get_idx(i, j):
            if i > j: i, j = j, i
            return i * (n - 1) - i * (i - 1) // 2 + j - i - 1
            
        # Build adjacency lists per color
        adj = [set() for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[get_idx(i, j)]
                adj[c].add((i, j))
                adj[c].add((j, i))
        
        # Check for C4 in each color
        for c in range(5):
            # For each pair of vertices (u, v), count common neighbors
            # If count >= 2, C4 exists
            # Optimization: only check pairs that are NOT connected?
            # No, C4 is u-a-v-b-u. u and v are opposite.
            # If u and v are connected, it's a triangle + edge, not a C4.
            # So we only need to check pairs (u, v) that are NOT edges in color c.
            # Actually, the definition of C4 in a graph usually implies induced or just subgraph?
            # "No two vertices with two common neighbours in one colour".
            # This applies to ANY two vertices.
            # If u and v are connected in color c, and they have 2 common neighbors, 
            # then u-a-v-b-u is a C4.
            # So we must check all pairs (u, v).
            
            adj_list = [set() for _ in range(n)]
            for (u, v) in adj[c]:
                adj_list[u].add(v)
                adj_list[v].add(u)
                
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj_list[u] & adj_list[v]
                    if len(common) >= 2:
                        return False
        return True

    def try_build(target_n, seed):
        random.seed(seed)
        neighbors = [set() for _ in range(5)]
        
        def add_vertex(n):
            # n is the new vertex index (0-indexed, so we are adding vertex n)
            # It connects to 0..n-1
            new_neighbors = [set() for _ in range(5)]
            
            for i in range(n):
                valid_colors = []
                for c in range(5):
                    # Check if i can be connected to n with color c
                    # Condition: len(neighbors[c][i] & new_neighbors[c]) < 2
                    common = neighbors[c][i] & new_neighbors[c]
                    if len(common) < 2:
                        valid_colors.append((len(common), c))
                        
                if not valid_colors:
                    return False
                
                # Heuristic: pick color with min common neighbors
                # If tie, random
                min_common = min(x[0] for x in valid_colors)
                candidates = [x[1] for x in valid_colors if x[0] == min_common]
                c = random.choice(candidates)
                
                # Update structures
                neighbors[c][i].add(n)
                new_neighbors[c].add(i)
                
            # After all edges added, update global neighbors
            for c in range(5):
                for v in new_neighbors[c]:
                    neighbors[c][v].add(n)
                    neighbors[c][n].add(v)
                    
            return True

        current_n = 1
        while current_n < target_n:
            if add_vertex(current_n):
                current_n += 1
            else:
                break
                
        if current_n > best_n:
            # Extract colors
            colors = []
            for i in range(current_n):
                for j in range(i + 1, current_n):
                    # Find color
                    for c in range(5):
                        if j in neighbors[c][i]:
                            colors.append(c)
                            break
            return current_n, colors

    # Run multiple restarts
    seed = 42
    max_target = 30 # We know 26 is possible, try up to 29
    
    while time.time() < deadline:
        n_found, colors_found = try_build(max_target, seed)
        if n_found > best_n:
            best_n = n_found
            best_colors = colors_found
            # If we hit 26 or higher, we are very good. 
            # We can try to push further or just return if time is tight.
            if best_n >= 26:
                # Try to improve slightly or break if time is low
                if time.time() > deadline - 20:
                    break
        seed += 1
        
    return {'n': best_n, 'colors': best_colors}
