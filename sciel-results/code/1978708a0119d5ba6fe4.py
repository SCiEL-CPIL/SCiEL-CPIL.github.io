import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Precompute edge indices
    # We will try to find a valid coloring for n=17, then 16, etc.
    # If we find one, we return it.
    
    best_n = 9
    best_colors = [min(j - i, 17 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    # Actually, let's use the seed's logic for n=9 just in case, but the seed logic was for n=9 specifically.
    # Seed: n=9, colors = [min(j - i, n - j + i) - 1 ...]
    # Let's just hardcode the n=9 seed result as fallback.
    seed_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    best_colors = seed_colors
    best_n = 9

    # Try to find a valid coloring for n from 17 down to 10
    for target_n in range(17, 9, -1):
        if time.time() > deadline - 5:
            break
        
        result = search_for_coloring(target_n, deadline)
        if result is not None:
            best_n = target_n
            best_colors = result
            # If we found 17, we can stop as it's the known best (or close to it)
            if target_n >= 17:
                break
        else:
            # If we didn't find one for this n, we might still have found a partial one?
            # No, search_for_coloring returns None if it didn't complete a valid coloring.
            # We could potentially return the largest partial valid coloring, but the task asks for a valid K_n coloring.
            # So we just continue to smaller n.
            pass

    return {'n': best_n, 'colors': best_colors}

def search_for_coloring(n, deadline):
    num_edges = n * (n - 1) // 2
    # Map (i, j) to index in flat array
    # Lexicographic order: (0,1), (0,2)...(0,n-1), (1,2)...
    idx_map = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx_map[(i, j)] = idx
            idx += 1
    
    # colors array: -1 means unassigned
    colors = [-1] * num_edges
    
    # Adjacency lists for each color, updated dynamically
    # adj[c][u] = set of neighbors v such that (u,v) is color c
    adj = [ [set() for _ in range(n)] for _ in range(4) ]
    
    # Order of edges to assign. 
    # Heuristic: assign edges involving lower indices first? Or random?
    # Randomization helps escape local minima.
    # Let's try a semi-random order: shuffle the edges, but maybe keep some structure.
    # Actually, just shuffling all edges is a good start for randomized backtracking.
    edge_order = list(range(num_edges))
    random.shuffle(edge_order)
    
    # Reverse mapping: index -> (i, j)
    edge_info = [None] * num_edges
    for (i, j), idx in idx_map.items():
        edge_info[idx] = (i, j)
        
    # Backtracking function
    # To avoid recursion depth issues and overhead, we can use a stack or just recursion with sys.setrecursionlimit.
    # For n=17, num_edges=136. Depth 136 is fine.
    
    def check_c6_conflict(u, v, c):
        """
        Check if adding edge (u, v) with color c creates a monochromatic C6.
        A C6 through (u, v) has form u-v-w-x-y-z-u.
        Edges (v,w), (w,x), (x,y), (y,z), (z,u) must be color c.
        This is equivalent to checking if there is a path of length 4 from v to u 
        in the current graph of color c (excluding the edge (u,v) itself, which we are adding).
        """
        # We need to find w, x, y, z such that:
        # 1. (v, w) is color c
        # 2. (w, x) is color c
        # 3. (x, y) is color c
        # 4. (y, z) is color c
        # 5. (z, u) is color c
        # And all vertices u, v, w, x, y, z are distinct.
        
        # Iterate over neighbors of v in color c
        for w in adj[c][v]:
            if w == u: continue # (v,u) is the edge we are adding, not in adj yet
            # w is distinct from u, v
            for x in adj[c][w]:
                if x == v or x == u: continue
                # x distinct from v, w, u
                for y in adj[c][x]:
                    if y == w or y == u or y == v: continue
                    # y distinct from x, w, u, v
                    for z in adj[c][y]:
                        if z == x or z == u or z == v or z == w: continue
                        # z distinct from y, x, w, u, v
                        # Check if (z, u) is color c
                        if u in adj[c][z]:
                            return True
        return False

    def assign(idx, c):
        i, j = edge_info[idx]
        # Ensure i < j for consistency, though idx_map was built with i<j
        # adj is symmetric
        adj[c][i].add(j)
        adj[c][j].add(i)
        colors[idx] = c

    def unassign(idx, c):
        i, j = edge_info[idx]
        adj[c][i].remove(j)
        adj[c][j].remove(i)
        colors[idx] = -1

    def backtrack(step):
        nonlocal best_colors_found, found
        if found:
            return True
        if time.time() > deadline:
            return False
            
        if step == num_edges:
            found = True
            # Copy the colors
            best_colors_found = colors[:]
            return True
            
        idx = edge_order[step]
        i, j = edge_info[idx]
        
        # Try colors 0, 1, 2, 3 in random order
        colors_to_try = [0, 1, 2, 3]
        random.shuffle(colors_to_try)
        
        for c in colors_to_try:
            # Check if assigning color c to (i, j) creates a C6
            if not check_c6_conflict(i, j, c):
                assign(idx, c)
                if backtrack(step + 1):
                    return True
                unassign(idx, c)
                
        return False

    found = False
    best_colors_found = None
    
    # Run backtracking
    backtrack(0)
    
    if found:
        return best_colors_found
    else:
        return None

# EVOLVE-BLOCK-END
