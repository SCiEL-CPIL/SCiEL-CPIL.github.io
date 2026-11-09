import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 140.0

    # 1. Generate n=32 base coloring
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    colors_32 = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors_32.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    
    # Initialize matrix for max n=50
    max_n = 50
    color_matrix = np.zeros((max_n, max_n), dtype=np.uint8)
    
    # Fill base 32
    idx = 0
    for i in range(32):
        for j in range(i + 1, 32):
            color_matrix[i, j] = colors_32[idx]
            color_matrix[j, i] = colors_32[idx]
            idx += 1

    current_n = 32

    # Helper to solve CSP for adding a new vertex
    def solve_csp_for_new_vertex(n_old):
        """
        Finds colors for edges (n_old, u) for u in 0..n_old-1
        such that no mono triangle is formed.
        Returns list of colors or None.
        """
        # Constraints: For each u < w in 0..n_old-1, if color_matrix[u,w] == k,
        # then NOT (x_u == k and x_w == k)
        
        # Precompute constraints: for each variable u, list of (w, k) where k is color(u,w)
        # Actually, we just need to check validity during backtracking.
        
        # Variable ordering: Random or by degree?
        # Let's try random restarts with backtracking.
        
        vars = list(range(n_old))
        random.shuffle(vars)
        
        assignment = {}
        
        def is_valid(u, c):
            # Check if assigning color c to u conflicts with any previously assigned v
            for v, cv in assignment.items():
                if v == u: continue
                edge_color = color_matrix[u, v]
                # Conflict if edge_color == c and edge_color == cv
                if edge_color == c and edge_color == cv:
                    return False
            return True

        def backtrack(idx):
            if idx == len(vars):
                return True
            u = vars[idx]
            # Try colors 0-3
            # Heuristic: try colors that appear less often in constraints?
            # Just try 0,1,2,3
            for c in range(4):
                if is_valid(u, c):
                    assignment[u] = c
                    if backtrack(idx + 1):
                        return True
                    del assignment[u]
            return False

        # Try a few random orderings
        for attempt in range(20):
            random.shuffle(vars)
            assignment.clear()
            if backtrack(0):
                # Convert assignment to list
                result = [0] * n_old
                for u, c in assignment.items():
                    result[u] = c
                return result
        return None

    # Growth loop
    for target_n in range(33, max_n + 1):
        if time.time() - start_time >= time_limit - 5:
            break
            
        n_old = target_n - 1
        # Try to find valid colors for the new vertex (index n_old)
        # We might need to try multiple times if the first random shuffle fails?
        # The CSP is deterministic given the matrix, but the solver is randomized.
        
        found = False
        # Give each vertex addition a small time budget
        vertex_deadline = time.time() + 2.0
        
        while time.time() < vertex_deadline:
            colors_new = solve_csp_for_new_vertex(n_old)
            if colors_new is not None:
                # Update matrix
                for u in range(n_old):
                    color_matrix[n_old, u] = colors_new[u]
                    color_matrix[u, n_old] = colors_new[u]
                found = True
                break
                
        if not found:
            # If we can't extend, we stop
            # We could try to modify the existing coloring?
            # For now, just break.
            break
        
        current_n = target_n
        
    # Return result
    n = current_n
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(int(color_matrix[i, j]))
            
    return {'n': n, 'colors': colors}
