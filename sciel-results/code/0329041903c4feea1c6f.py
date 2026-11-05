import random

def construct():
    # Generate Z_17 coloring
    n = 16
    p = 17
    # Edge list in lexicographic order
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            
    # Initial coloring: (i+j) % 17
    # Map edge (i,j) to index
    # We need to return a list of 120 ints.
    # Let's store colors in a 2D array for fast access
    cols = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            c = (i + j) % p
            cols[i][j] = c
            cols[j][i] = c
            
    # Precompute K4s
    k4s = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    # Indices of edges: (i,j), (i,k), (i,l), (j,k), (j,l), (k,l)
                    k4s.append(((i,j), (i,k), (i,l), (j,k), (j,l), (k,l)))
                    
    def is_valid(cols):
        for e1, e2, e3, e4, e5, e6 in k4s:
            c1 = cols[e1[0]][e1[1]]
            c2 = cols[e2[0]][e2[1]]
            c3 = cols[e3[0]][e3[1]]
            c4 = cols[e4[0]][e4[1]]
            c5 = cols[e5[0]][e5[1]]
            c6 = cols[e6[0]][e6[1]]
            if len({c1, c2, c3, c4, c5, c6}) < 5:
                return False
        return True

    # Check initial validity
    if not is_valid(cols):
        # Fallback to random or previous best
        # Z_17 should be valid
        pass

    # Greedy reduction
    current_max = p - 1
    # We want to minimize the number of colors.
    # The colors used are 0..current_max.
    # We try to remove the highest color index.
    
    # To make it more robust, we can try to merge any color c1 into c2.
    # But removing the highest is a good heuristic.
    
    # Let's do a few passes.
    for _ in range(10): # Max 10 reduction steps
        reduced = False
        # Try to remove color 'current_max'
        c_remove = current_max
        # Try to map it to any other color 0..current_max-1
        # To be systematic, try all targets.
        found = False
        for c_target in range(current_max):
            # Save state
            orig_cols = [row[:] for row in cols]
            
            # Apply change
            for i in range(n):
                for j in range(i + 1, n):
                    if cols[i][j] == c_remove:
                        cols[i][j] = c_target
                        cols[j][i] = c_target
                        
            if is_valid(cols):
                found = True
                reduced = True
                break
            else:
                # Restore
                for i in range(n):
                    for j in range(i + 1, n):
                        cols[i][j] = orig_cols[i][j]
                        cols[j][i] = orig_cols[j][i]
                        
        if not found:
            # Try removing other colors?
            # For simplicity, if we can't remove the highest, we stop.
            # Or we can try to remove a random color.
            # Let's try to remove a random color from the used set.
            used_colors = set()
            for i in range(n):
                for j in range(i + 1, n):
                    used_colors.add(cols[i][j])
                    
            # Shuffle used colors
            clist = list(used_colors)
            random.shuffle(clist)
            
            for c_remove in clist:
                orig_cols = [row[:] for row in cols]
                
                # Pick a target color, preferably one that is already used and "close"
                # Just pick the smallest used color
                c_target = min(used_colors)
                if c_target == c_remove:
                    continue
                    
                for i in range(n):
                    for j in range(i + 1, n):
                        if cols[i][j] == c_remove:
                            cols[i][j] = c_target
                            cols[j][i] = c_target
                            
                if is_valid(cols):
                    reduced = True
                    break
                else:
                    for i in range(n):
                        for j in range(i + 1, n):
                            cols[i][j] = orig_cols[i][j]
                            cols[j][i] = orig_cols[j][i]
                
                if reduced:
                    break

        if not reduced:
            break
            
    # Return the final coloring
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            result.append(cols[i][j])
            
    return result
