"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Build the 32-vertex base coloring (Greenwood-Gleason: two copies of 16-vertex Clebsch)
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_map, x = {}, 1
    for e in range(15):
        log_map[x] = e
        x = mul(x, 2)

    def base_colors(n=32):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if i // 16 == j // 16:
                    colors.append(log_map[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return colors

    def color_index(n, i, j):
        """Get index in colors array for edge (i,j), i<j in lexicographic order."""
        if i > j:
            i, j = j, i
        return i * (2*n - i - 1) // 2 + (j - i - 1)

    # Start with 32-vertex base
    n = 32
    colors = base_colors(n)
    best_n = n
    best_colors = colors[:]

    # Build adjacency as bitmasks for speed: adj_mask[v][c] is a bitmask of vertices connected to v by color c
    def build_adj_masks(colors, n):
        adj_masks = [[0]*4 for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                c = colors[color_index(n, i, j)]
                adj_masks[i][c] |= (1 << j)
                adj_masks[j][c] |= (1 << i)
        return adj_masks

    def try_add_vertex_bitmask(n, adj_masks, deadline):
        """
        Try to add vertex n. Use bitmask-based backtracking with forward checking.
        assignment[i] = color assigned to edge (new_vertex, i)
        Constraint: for each color c, the set {i : assignment[i]==c} must be independent in G_c.
        Equivalent: if assignment[i]==c and assignment[j]==c, then bit i is NOT set in adj_masks[j][c].
        """
        if time.time() > deadline:
            return None

        assignment = [-1] * n
        
        # For forward checking: maintain for each color c, the set of vertices already assigned to c
        # A vertex i can be assigned color c only if no vertex j in assigned_set[c] has i in adj_masks[j][c]
        
        def backtrack(pos):
            nonlocal assignment
            if time.time() > deadline:
                return False
            if pos == n:
                return True
            
            # Try colors in a heuristic order: prefer colors with fewer conflicts
            # First, determine which colors are still feasible for vertex pos
            feasible_colors = []
            for c in range(4):
                # Check: is pos compatible with all vertices already assigned color c?
                # We need: for all j < pos with assignment[j]==c, pos is NOT in adj_masks[j][c]
                # Equivalently: the bitmask of vertices assigned color c that are adjacent to pos via color c must be 0
                # But we don't have a direct "assigned to c" bitmask. Let's check manually.
                ok = True
                for j in range(pos):
                    if assignment[j] == c:
                        if (adj_masks[pos][c] >> j) & 1:
                            ok = False
                            break
                if ok:
                    feasible_colors.append(c)
            
            if not feasible_colors:
                return False
            
            # Order by least constraining value heuristic: try colors that leave more options
            # For simplicity, try in order but with a random tie-break
            random.shuffle(feasible_colors)
            
            for c in feasible_colors:
                assignment[pos] = c
                if backtrack(pos + 1):
                    return True
                assignment[pos] = -1
            
            return False

        if backtrack(0):
            return assignment[:]
        return None

    # Main loop: try to extend the coloring
    adj_masks = build_adj_masks(colors, n)
    
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Try to add a vertex
        assignment = try_add_vertex_bitmask(n, adj_masks, deadline)
        
        if assignment is not None:
            # Success! Add the new vertex
            new_n = n + 1
            new_colors = colors[:]
            for i in range(n):
                new_colors.append(assignment[i])
            
            # Update adjacency masks
            adj_masks = [[0]*4 for _ in range(new_n)]
            for i in range(new_n):
                for j in range(i+1, new_n):
                    if i < n and j < n:
                        c = colors[color_index(n, i, j)]
                    elif i == new_n - 1:
                        c = assignment[j]
                    else:
                        c = assignment[i]
                    adj_masks[i][c] |= (1 << j)
                    adj_masks[j][c] |= (1 << i)
            
            colors = new_colors
            n = new_n
            best_n = max(best_n, n)
            best_colors = colors[:]
        else:
            # Stuck - try randomizing the order of vertices we assign
            # Reorder the problem: shuffle which vertices we consider first
            # This is complex, so just break for now
            break

    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
