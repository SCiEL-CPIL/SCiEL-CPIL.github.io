<<<<<<< SEARCH
import random

def construct():
    # Paley graph construction for n=29
    n_base = 29
    q = n_base
    # Quadratic residues mod 29
    squares = set()
    for x in range(1, q):
        squares.add((x * x) % q)
    
    colors_base = []
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if (j - i) % q in squares:
                colors_base.append(0)
            else:
                colors_base.append(1)
                
    # Verify base coloring
    if is_valid(n_base, colors_base):
        # Try to extend to n=30
        n_target = 30
        # We need to add 29 edges for the new vertex 29
        # Edges are (29, 0), (29, 1), ..., (29, 28)
        # In lexicographic order, these are the last 29 entries in the list of size C(30,2)=435
        
        # We will try to find a valid assignment for these 29 edges
        # We use a random restart strategy
        
        # To speed up, we can fix some edges to 0 to break symmetry?
        # The Paley graph is vertex transitive, so vertex 29 is equivalent to any other.
        # However, the existing graph is fixed.
        
        # Let's try 100 random assignments
        for _ in range(100):
            new_edges = [random.randint(0, 1) for _ in range(n_base)]
            # Construct full color list
            # The first C(29,2) = 406 entries are colors_base
            # The last 29 entries are new_edges
            # Wait, lexicographic order:
            # Pairs (i,j) with i<j.
            # For n=30, pairs involving 29 are (0,29), (1,29), ..., (28,29).
            # These are NOT the last 29 in a simple concatenation if we just append.
            # Let's check the order.
            # i=0: (0,1)...(0,28), (0,29)
            # i=1: (1,2)...(1,28), (1,29)
            # ...
            # i=28: (28,29)
            # So the edges involving 29 are scattered.
            
            # It is easier to build the full list from scratch or map indices.
            # Let's build a helper to get color for (i,j) from a list.
            # Or just generate the full list.
            
            full_colors = [0] * (30 * 29 // 2) # 435
            
            # Fill base part
            idx = 0
            for i in range(n_base):
                for j in range(i + 1, n_base):
                    full_colors[idx] = colors_base[idx]
                    idx += 1
            # This is wrong. The base colors are for n=29.
            # The indices for n=30 are different.
            
            # Let's just generate the full list directly in the loop.
            full_colors = []
            for i in range(30):
                for j in range(i + 1, 30):
                    if i < n_base and j < n_base:
                        # Use base coloring
                        # Need to map (i,j) in n=29 to index in colors_base
                        # Index in n=29: sum_{k=0}^{i-1} (29-1-k) + (j-i-1)
                        # Or simpler: just recompute? No, use the list.
                        # Let's precompute a mapping or just re-derive from squares.
                        if (j - i) % q in squares:
                            full_colors.append(0)
                        else:
                            full_colors.append(1)
                    else:
                        # One of i or j is 29
                        # Since i < j, j must be 29
                        # Edge is (i, 29)
                        # i ranges from 0 to 28
                        # We stored new_edges[i] for edge (i, 29)
                        full_colors.append(new_edges[i])
            
            if is_valid(30, full_colors):
                return {'n': 30, 'colors': full_colors}
                
        # If no extension found, return base
        return {'n': n_base, 'colors': colors_base}
    else:
        # Fallback to n=17 if 29 fails (should not happen)
        squares_17 = {x * x % 17 for x in range(1, 17)}
        return {'n': 17, 'colors': [0 if (j - i) % 17 in squares_17 else 1 for i in range(17) for j in range(i + 1, 17)]}
=======
import time

def construct():
    """
    Try to find a valid 2-coloring of K_30 with no monochromatic K_{2,2,2}.
    Strategy:
    1. Start with the Paley graph on 29 vertices (known to be K_{2,2,2}-free in both colors).
    2. Use backtracking to assign colors to the 29 edges incident to vertex 29.
    3. Use aggressive pruning: after each assignment, check if either color class
       now contains a K_{2,2,2}. If so, backtrack.
    4. Symmetry breaking: fix edge (0,29) = 0 (since we can swap colors globally).
    5. If backtracking fails within time limit, fall back to n=29.
    """
    n_base = 29
    n_target = 30
    q = n_base
    squares = set()
    for x in range(1, q):
        squares.add((x * x) % q)
    
    # Precompute base colors for edges among vertices 0..28
    base_color = [[-1] * n_base for _ in range(n_base)]
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if (j - i) % q in squares:
                base_color[i][j] = base_color[j][i] = 0
            else:
                base_color[i][j] = base_color[j][i] = 1
    
    # Verify base coloring is valid
    if not _is_valid_partial(n_base, base_color):
        # Fallback to smaller n
        n_small = 17
        squares_small = {x * x % n_small for x in range(1, n_small)}
        colors_small = []
        for i in range(n_small):
            for j in range(i + 1, n_small):
                if (j - i) % n_small in squares_small:
                    colors_small.append(0)
                else:
                    colors_small.append(1)
        return {'n': n_small, 'colors': colors_small}
    
    # Backtracking search for vertex 29's edges
    # Edge (i, 29) for i in 0..28
    # We'll use a recursive backtracking with pruning
    
    start_time = time.time()
    time_limit = 115  # seconds, leave buffer for overhead
    
    # new_edge_color[i] = color of edge (i, 29), -1 means unassigned
    new_edge_color = [-1] * n_base
    
    # Precompute for each color, the adjacency among vertices 0..28
    # adj_color[c][u][v] = True if edge (u,v) has color c
    adj0 = [[False] * n_base for _ in range(n_base)]
    adj1 = [[False] * n_base for _ in range(n_base)]
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if base_color[i][j] == 0:
                adj0[i][j] = adj0[j][i] = True
            else:
                adj1[i][j] = adj1[j][i] = True
    
    # For fast K_{2,2,2} checking with the new vertex, we need to track
    # which vertices are connected to vertex 29 in each color.
    # Let's use bitmasks for efficiency.
    # n_base = 29, so we can use Python ints as bitmasks.
    
    # For each color c, let S_c be the set of vertices (among 0..28) connected to 29 via color c.
    # A K_{2,2,2} involving vertex 29 must have 29 in one part, and 2 vertices from each
    # of the other two parts, all connected to 29 via the same color, and forming K_{2,2,2}
    # with 29.
    
    # More precisely: if 29 is in part A with vertex u, then we need two more vertices v, w
    # such that:
    # - 29-u, 29-v, 29-w all same color c
    # - u-v, u-w, v-w... wait no.
    
    # K_{2,2,2} has parts A={a1,a2}, B={b1,b2}, C={c1,c2}
    # All edges between different parts exist.
    # If 29 is in A, say 29=a1, then a2 must be connected to all of B and C.
    # And 29 must be connected to all of B and C.
    
    # So for a fixed color c, if we have S_c = {vertices connected to 29 via color c},
    # then a K_{2,2,2} in color c involving 29 exists iff there exist:
    # - a2 in S_c
    # - B = {b1, b2} subset of S_c, b1 != b2, b1 != a2, b2 != a2
    # - C = {c1, c2} subset of S_c, c1 != c2, c1 != a2, c1 != b1, c1 != b2, etc.
    # such that:
    # - a2 is connected to b1, b2, c1, c2 via color c
    # - b1, b2 are connected to c1, c2 via color c
    # - (b1-b2 and c1-c2 edges don't matter, they're in same part)
    
    # This is complex. Let's just use the general K_{2,2,2} checker but optimized.
    
    def backtrack(pos, S0, S1):
        """
        pos: next vertex index to assign (0..28) for edge (pos, 29)
        S0: bitmask of vertices connected to 29 via color 0 (among 0..28)
        S1: bitmask of vertices connected to 29 via color 1 (among 0..28)
        Returns: True if solution found, False otherwise
        """
        if time.time() - start_time > time_limit:
            return False
        
        if pos == n_base:
            # All edges assigned. Check if valid.
            # Build full adjacency and check
            return _check_full(adj0, adj1, S0, S1)
        
        # Try color 0 for edge (pos, 29)
        new_S0 = S0 | (1 << pos)
        new_S1 = S1
        new_edge_color[pos] = 0
        if _prune_check(adj0, adj1, new_S0, new_S1):
            if backtrack(pos + 1, new_S0, new_S1):
                return True
        new_edge_color[pos] = -1
        
        # Try color 1 for edge (pos, 29)
        new_S0 = S0
        new_S1 = S1 | (1 << pos)
        new_edge_color[pos] = 1
        if _prune_check(adj0, adj1, new_S0, new_S1):
            if backtrack(pos + 1, new_S0, new_S1):
                return True
        new_edge_color[pos] = -1
        
        return False
    
    # Symmetry breaking: fix edge (0, 29) = 0
    S0_init = 1 << 0  # vertex 0 connected to 29 via color 0
    S1_init = 0
    new_edge_color[0] = 0
    
    found = backtrack(1, S0_init, S1_init)
    
    if found:
        # Build the full color list for n=30
        full_colors = []
        for i in range(n_target):
            for j in range(i + 1, n_target):
                if i < n_base and j < n_base:
                    full_colors.append(base_color[i][j])
                else:
                    # j == 29, i in 0..28
                    full_colors.append(new_edge_color[i])
        return {'n': n_target, 'colors': full_colors}
    
    # Fallback to n=29
    colors_base = []
    for i in range(n_base):
        for j in range(i + 1, n_base):
            colors_base.append(base_color[i][j])
    return {'n': n_base, 'colors': colors_base}
>>>>>>> REPLACE

