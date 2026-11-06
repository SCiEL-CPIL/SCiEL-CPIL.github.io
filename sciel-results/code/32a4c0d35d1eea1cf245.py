import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Try to find the largest n for which a valid 5-coloring exists.
    # We know n=26 is possible. We will try to construct it using a greedy algorithm.
    # If greedy fails, we try smaller n.
    
    # Fallback result if everything fails (should not happen for n>=11)
    fallback_n = 11
    fallback_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    target_ns = [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]
    
    for n in target_ns:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random restarts for the greedy algorithm
        found = False
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            colors = greedy_construct(n)
            if colors is not None:
                # Verify the solution
                if is_valid(n, colors):
                    return {'n': n, 'colors': colors}
                else:
                    # Should not happen if greedy is correct, but just in case
                    pass
        
    return {'n': fallback_n, 'colors': fallback_colors}

def is_valid(n, colors):
    """
    Verify that the coloring has no monochromatic C4.
    colors is a list of length n*(n-1)//2 in lexicographic order.
    """
    # Reconstruct adjacency lists per color
    # neighbors[c][v] is a set of neighbors of v in color c
    neighbors = [set() for _ in range(n)]
    
    # Map edge index to (i, j)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            neighbors[i].add((j, c))
            neighbors[j].add((i, c))
            idx += 1
    
    # Check for C4
    # A C4 in color c exists if there are two vertices u, v that share two neighbors in color c.
    # Equivalently, for each color c, the graph must be C4-free.
    # We can check this by iterating over pairs of vertices and checking common neighbors.
    # For n=26, this is fast enough.
    
    for c in range(5):
        # Get neighbors in color c for each vertex
        # To speed up, we can just check all pairs (u, v)
        for u in range(n):
            for v in range(u + 1, n):
                common_count = 0
                for w in range(n):
                    if w == u or w == v:
                        continue
                    # Check if edge (u, w) is color c and (v, w) is color c
                    # We need to find the index of (u, w) and (v, w)
                    # This is O(n^3) which is 26^3 = 17576 operations. Very fast.
                    # But we need to map (u, w) to index.
                    pass
                
                # Let's use a more direct check with precomputed indices
                # Actually, let's just use the bitsets or sets properly.
                
    # Let's implement a faster verification using bitsets or sets.
    # neighbors_color[c][v] = set of neighbors of v in color c
    neighbors_color = [ [set() for _ in range(n)] for _ in range(5) ]
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            neighbors_color[c][i].add(j)
            neighbors_color[c][j].add(i)
            idx += 1
            
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                common = neighbors_color[c][u].intersection(neighbors_color[c][v])
                if len(common) >= 2:
                    return False
    return True

def greedy_construct(n):
    """
    Attempt to construct a valid 5-coloring of K_n using a greedy approach.
    Returns None if it fails (gets stuck).
    """
    # We will add vertices one by one.
    # State:
    # colors: list of colors for edges. We build it incrementally.
    # To make it easier, we can store the color of edge (i, j) in a 2D array or dict.
    # Let's use a 2D list `color_matrix` of size n x n, initialized to -1.
    color_matrix = [[-1] * n for _ in range(n)]
    
    # To check if adding edge (u, v) with color c creates a C4:
    # We need to check if there exists w such that (u, w) is color c and (v, w) is color c.
    # If such w exists, then u-v-w-u' ... wait.
    # A C4 is u-v-w-x-u.
    # If we add edge (u, v) with color c, it creates a C4 if there is already a path of length 2 between u and v in color c.
    # i.e., there exists w such that color_matrix[u][w] == c and color_matrix[w][v] == c.
    # If so, u-w-v is a path of length 2. Adding u-v closes the cycle u-w-v-u? No, that's a triangle.
    # A C4 is u-v-w-x-u.
    # If we add edge (u, v), it completes a C4 if there is a path u-w-x-v of length 3? No.
    # It completes a C4 if there is a path u-w-v of length 2? No, that's a triangle.
    # It completes a C4 if there is a path u-w-x-v of length 3?
    # Let's trace:
    # Existing edges: (u, w) color c, (w, x) color c, (x, v) color c.
    # Add (u, v) color c.
    # Cycle: u-w-x-v-u. Length 4. Yes.
    # So, adding (u, v) with color c is forbidden if there exists a path of length 3 between u and v in color c.
    # Equivalently, if there exists x such that (u, x) is color c and (x, v) is color c?
    # If (u, x) is c and (x, v) is c, then u-x-v is a path of length 2.
    # Adding (u, v) creates a triangle u-x-v-u.
    # Triangles are allowed? The constraint is "no monochromatic four-cycles".
    # So triangles are fine.
    # So we only forbid if there is a path of length 3.
    
    # Wait, let's re-read the definition of C4-free.
    # "No two vertices with two common neighbours in one colour".
    # If u and v have two common neighbours w and x in color c, then:
    # (u, w) is c, (v, w) is c, (u, x) is c, (v, x) is c.
    # This forms a C4: u-w-v-x-u.
    # So the condition is: for any pair (u, v), the number of common neighbours in color c must be < 2.
    
    # So when adding edge (u, v) with color c:
    # We must ensure that u and v do not already have 2 or more common neighbours in color c.
    # Common neighbours of u and v in color c are vertices w such that (u, w) is c and (v, w) is c.
    # Note: w cannot be u or v.
    
    # So the check is:
    # count = 0
    # for w in range(n):
    #   if w != u and w != v:
    #     if color_matrix[u][w] == c and color_matrix[v][w] == c:
    #       count += 1
    # if count >= 1:
    #   # Then adding (u, v) would make the total common neighbours >= 2 (the existing ones + ?)
    #   # Wait. The existing common neighbours are already there.
    #   # If there is ALREADY 1 common neighbour w, then adding (u, v) does NOT change the common neighbours of u and v.
    #   # The common neighbours of u and v are determined by the edges (u, w) and (v, w).
    #   # The edge (u, v) itself is not a common neighbour.
    #   # So adding (u, v) does NOT create a new common neighbour for u and v.
    #   # It only creates a C4 if there is ALREADY a path of length 2? No.
    #   # If there is ALREADY a common neighbour w, then u-w-v is a path of length 2.
    #   # Does this form a C4? No, it's a triangle if (u, v) exists.
    #   # But we are ADDING (u, v).
    #   # So if there is already a common neighbour w, then we have u-w-v.
    #   # Adding u-v creates a triangle u-w-v-u.
    #   # This is NOT a C4.
    #   
    #   # So when does adding (u, v) create a C4?
    #   # It creates a C4 if there is a path of length 3 between u and v in color c.
    #   # i.e., u-w-x-v.
    #   # Then u-v-w-x-u is a C4.
    #   
    #   # Let's re-evaluate "two common neighbours".
    #   # If u and v have two common neighbours w and x.
    #   # Then (u, w), (v, w), (u, x), (v, x) are all color c.
    #   # This forms the cycle u-w-v-x-u.
    #   # This cycle exists REGARDLESS of whether (u, v) is an edge or what color it is.
    #   # Wait. The cycle u-w-v-x-u uses edges (u, w), (w, v), (v, x), (x, u).
    #   # These are 4 distinct edges.
    #   # If (u, v) is also an edge, it's a chord, but the C4 still exists.
    #   # So the condition "no two vertices with two common neighbours" is the correct one to check.
    #   
    #   # So, when adding edge (u, v) with color c:
    #   # Does it change the number of common neighbours of any pair?
    #   # It adds (u, v) to the adjacency list of u and v.
    #   # So for any other vertex w:
    #   #   Does w become a common neighbour of u and some other vertex?
    #   #   Common neighbours of u and x: vertices y such that (u, y) is c and (x, y) is c.
    #   #   If we add (u, v) with color c, then v becomes a neighbour of u in color c.
    #   #   So if (x, v) is also color c, then v becomes a NEW common neighbour of u and x.
    #   #   So we must check if adding (u, v) causes ANY pair to have 2 common neighbours.
    #   
    #   # Specifically:
    #   # 1. Pair (u, v): Their common neighbours are unchanged by adding (u, v).
    #   #    So if they already had 2 common neighbours, the graph was already invalid.
    #   #    We assume the previous state was valid. So they have < 2 common neighbours.
    #   #    Adding (u, v) doesn't change this. So (u, v) is safe regarding their own common neighbours.
    #   
    #   # 2. Pair (u, x) for x != v:
    #   #    New common neighbour candidate: v.
    #   #    v is a common neighbour of u and x if (u, v) is c and (x, v) is c.
    #   #    We are setting (u, v) to c.
    #   #    So if (x, v) is c, then v becomes a common neighbour of u and x.
    #   #    We must ensure that u and x do not ALREADY have another common neighbour in color c.
    #   #    If they do, adding v as a second common neighbour creates a C4.
    #   
    #   # 3. Pair (v, x) for x != u:
    #   #    Symmetric to 2.
    #   
    #   # So the check is:
    #   # For all x in {0..n-1} \ {u, v}:
    #   #   If color_matrix[x][v] == c:
    #   #      # v is a common neighbour of u and x in color c (since we just set u-v to c)
    #   #      # Check if u and x have any OTHER common neighbour in color c.
    #   #      # If yes, invalid.
    #   #   If color_matrix[u][x] == c:
    #   #      # u is a common neighbour of v and x in color c (since we just set u-v to c)
    #   #      # Check if v and x have any OTHER common neighbour in color c.
    #   #      # If yes, invalid.
    
    # This is O(n^2) per edge addition.
    # Total edges ~ n^2/2.
    # Total complexity O(n^4). For n=26, 26^4 = 456,976. Very fast.
    
    # We will add edges in lexicographic order.
    # To allow backtracking, we would need to store state.
    # But let's first try a simple greedy without backtracking.
    # If it fails, we can try randomizing the order or colors.
    
    # Actually, a simple greedy in lexicographic order might get stuck.
    # Let's add a simple backtracking if we get stuck.
    # Or just try multiple random restarts with different color preferences.
    
    # For now, let's implement the simple greedy and see if it works for n=26.
    # If not, we will add backtracking.
    
    # To make it more robust, let's try colors in a random order for each edge.
    import random
    
    for i in range(n):
        for j in range(i + 1, n):
            # Try each color
            found_color = False
            # Shuffle colors to avoid bias
            colors_to_try = [0, 1, 2, 3, 4]
            random.shuffle(colors_to_try)
            
            for c in colors_to_try:
                if is_safe(color_matrix, i, j, c, n):
                    color_matrix[i][j] = c
                    color_matrix[j][i] = c
                    found_color = True
                    break
            
            if not found_color:
                return None # Failed
    
    # Convert matrix to list
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(color_matrix[i][j])
    return colors

def is_safe(color_matrix, u, v, c, n):
    """
    Check if adding edge (u, v) with color c is safe.
    Safe means it does not create a monochromatic C4.
    """
    # As derived above:
    # 1. Pair (u, v) is safe if they don't already have 2 common neighbours in c.
    #    But we assume previous state was valid, so they have < 2.
    #    Adding (u, v) doesn't change their common neighbours.
    #    So this check is actually unnecessary IF we maintain validity.
    #    However, it's good to be safe.
    #    Let's check it.
    common_uv = 0
    for w in range(n):
        if w != u and w != v:
            if color_matrix[u][w] == c and color_matrix[v][w] == c:
                common_uv += 1
                if common_uv >= 2:
                    return False # Already invalid? Or this prevents adding?
                    # If they already have 2, the graph is already invalid.
                    # But we assume it's valid. So this shouldn't happen.
    
    # 2. Check pairs (u, x) and (v, x) for x != u, v
    for x in range(n):
        if x == u or x == v:
            continue
        
        # Check if v becomes a new common neighbour of u and x
        if color_matrix[x][v] == c:
            # v is a common neighbour of u and x in color c
            # Check if u and x have any OTHER common neighbour in color c
            other_common = 0
            for w in range(n):
                if w == u or w == x or w == v:
                    continue
                if color_matrix[u][w] == c and color_matrix[x][w] == c:
                    other_common += 1
                    if other_common >= 1:
                        return False # u and x would have 2 common neighbours (v and w)
                        
        # Check if u becomes a new common neighbour of v and x
        if color_matrix[u][x] == c:
            # u is a common neighbour of v and x in color c
            # Check if v and x have any OTHER common neighbour in color c
            other_common = 0
            for w in range(n):
                if w == v or w == x or w == u:
                    continue
                if color_matrix[v][w] == c and color_matrix[x][w] == c:
                    other_common += 1
                    if other_common >= 1:
                        return False # v and x would have 2 common neighbours (u and w)
                        
    return True
