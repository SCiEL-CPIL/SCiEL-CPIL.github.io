import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 115.0  # 5s safety margin
    
    # 1. Generate Paley 29
    n_29 = 29
    squares_29 = {x * x % 29 for x in range(1, 29)}
    colors_29 = [0 if (j - i) % 29 in squares_29 else 1 for i in range(n_29) for j in range(i + 1, n_29)]
    
    # 2. Exact verification of Paley 29
    # n=29, C(29,6) ~ 4.9M tuples.
    # We can check this exactly.
    def check_valid_exact(colors, n):
        # Build adjacency matrix for O(1) access
        mat = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                mat[i][j] = c
                mat[j][i] = c
                idx += 1
        
        # Check for K2,2,2 in color 0 and 1
        # K2,2,2 is defined by 3 pairs of vertices (a,b), (c,d), (e,f)
        # such that all edges between pairs are of the same color.
        # Vertices: a,b,c,d,e,f distinct.
        # Edges: a-c, a-d, b-c, b-d, a-e, a-f, b-e, b-f, c-e, c-f, d-e, d-f
        # Total 12 edges.
        
        # To speed up, we can iterate over 6-tuples.
        # 4.9M iterations is okay in 110s?
        # 4.9M * 12 checks = 58M operations.
        # Python might take ~5-10 seconds.
        # This is acceptable.
        
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        for e in range(d+1, n):
                            for f in range(e+1, n):
                                # Check color 0
                                if (mat[a][c] == 0 and mat[a][d] == 0 and mat[b][c] == 0 and mat[b][d] == 0 and
                                    mat[a][e] == 0 and mat[a][f] == 0 and mat[b][e] == 0 and mat[b][f] == 0 and
                                    mat[c][e] == 0 and mat[c][f] == 0 and mat[d][e] == 0 and mat[d][f] == 0):
                                    return False
                                # Check color 1
                                if (mat[a][c] == 1 and mat[a][d] == 1 and mat[b][c] == 1 and mat[b][d] == 1 and
                                    mat[a][e] == 1 and mat[a][f] == 1 and mat[b][e] == 1 and mat[b][f] == 1 and
                                    mat[c][e] == 1 and mat[c][f] == 1 and mat[d][e] == 1 and mat[d][f] == 1):
                                    return False
        return True

    if check_valid_exact(colors_29, n_29):
        # We have a valid n=29. Store it.
        result_29 = {'n': 29, 'colors': colors_29}
    else:
        # Fallback if Paley 29 is somehow invalid (should not happen)
        # Use a smaller known valid graph, e.g., Paley 17
        n_17 = 17
        squares_17 = {x * x % 17 for x in range(1, 17)}
        colors_17 = [0 if (j - i) % 17 in squares_17 else 1 for i in range(n_17) for j in range(i + 1, n_17)]
        result_29 = {'n': 17, 'colors': colors_17}
        
    # 3. Try to find n=30
    # If we have time, try to extend or search for n=30.
    if time.time() < deadline - 5.0:
        n_30 = 30
        # Start with random coloring or extend Paley 29?
        # Extension is smarter.
        # Take colors_29 (29*28/2 = 406 edges).
        # Add vertex 29.
        # Edges (29, j) for j in 0..28.
        # Initialize randomly.
        colors_30 = colors_29[:]
        for j in range(29):
            colors_30.append(random.randint(0, 1))
            
        # Local Search: Find a violation, fix it.
        # A violation is a K2,2,2.
        # We need a fast way to find violations.
        # We can maintain a list of "bad" 6-tuples? Too expensive.
        # Instead, random sampling of 6-tuples to find a violation.
        # If found, flip a random edge in that violation.
        
        mat_30 = [[0]*30 for _ in range(30)]
        idx = 0
        for i in range(30):
            for j in range(i+1, 30):
                c = colors_30[idx]
                mat_30[i][j] = c
                mat_30[j][i] = c
                idx += 1
                
        def find_violation(mat, n):
            # Sample 1000 random 6-tuples
            for _ in range(1000):
                verts = random.sample(range(n), 6)
                a, b, c, d, e, f = verts
                # Check 0
                if (mat[a][c] == 0 and mat[a][d] == 0 and mat[b][c] == 0 and mat[b][d] == 0 and
                    mat[a][e] == 0 and mat[a][f] == 0 and mat[b][e] == 0 and mat[b][f] == 0 and
                    mat[c][e] == 0 and mat[c][f] == 0 and mat[d][e] == 0 and mat[d][f] == 0):
                    return (a, b, c, d, e, f, 0)
                # Check 1
                if (mat[a][c] == 1 and mat[a][d] == 1 and mat[b][c] == 1 and mat[b][d] == 1 and
                    mat[a][e] == 1 and mat[a][f] == 1 and mat[b][e] == 1 and mat[b][f] == 1 and
                    mat[c][e] == 1 and mat[c][f] == 1 and mat[d][e] == 1 and mat[d][f] == 1):
                    return (a, b, c, d, e, f, 1)
            return None

        current_colors = colors_30[:]
        current_mat = mat_30
        
        # Search loop
        while time.time() < deadline - 2.0:
            viol = find_violation(current_mat, 30)
            if viol is None:
                # No violation found in 1000 samples. 
                # Do a more thorough check? 
                # If we are lucky, it might be valid.
                # Let's do a small exact check on a subset?
                # Or just assume valid and verify at the end?
                # Better: Run exact check on the current state.
                # C(30,6) ~ 59M. Too slow for exact check every step.
                # But if sampler finds 0, it's likely valid.
                # Let's do one exact check here.
                if check_valid_exact(current_colors, 30):
                    return {'n': 30, 'colors': current_colors}
                else:
                    # Sampler missed a violation. 
                    # We need to find it.
                    # We can iterate through all 6-tuples to find one.
                    # This is slow, but we only do it once when sampler is "optimistic".
                    # Actually, if sampler is 0, we should just break and return 29?
                    # No, we can try to find the violation by exhaustive search in a limited way?
                    # If we can't find it fast, we might waste time.
                    # Let's just break and return 29.
                    break
            
            # Found violation
            a, b, c, d, e, f, col = viol
            # Pick a random edge from the 12 edges of the K2,2,2 to flip
            edges = [(a,c), (a,d), (b,c), (b,d), (a,e), (a,f), (b,e), (b,f), (c,e), (c,f), (d,e), (d,f)]
            u, v = random.choice(edges)
            if u > v: u, v = v, u
            
            # Flip
            old_c = current_mat[u][v]
            new_c = 1 - old_c
            current_mat[u][v] = new_c
            current_mat[v][u] = new_c
            
            # Update colors list
            # Index calculation:
            # idx = u*(2n - u - 1)//2 + (v - u - 1) ?
            # Simpler: precompute index map or just rebuild?
            # Rebuilding is O(n^2) = 900. Fast.
            idx = 0
            for i in range(30):
                for j in range(i+1, 30):
                    if i == u and j == v:
                        current_colors[idx] = new_c
                    idx += 1

    return result_29
