import time
import random
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0

    def build_seed_32():
        # GF(16) construction for n=32
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

        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                if i // 16 == j // 16:
                    colors.append(log[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return colors

    def extend_coloring(colors, n_current, n_target, time_limit=10.0):
        """
        Tries to extend a valid coloring of K_n_current to K_n_target.
        Uses randomized greedy with backtracking.
        """
        t_start = time.time()
        n = n_current
        # Convert colors to a 2D matrix for easy access
        # colors is a list of length n*(n-1)/2
        # We need to map (i,j) to index
        # idx = i * n - i*(i+1)//2 + (j - i - 1) is for fixed n.
        # As n grows, indices change. It's better to maintain a 2D array C.
        
        # Reconstruct C for current n
        C = np.zeros((n_target, n_target), dtype=np.int8)
        idx = 0
        for i in range(n_current):
            for j in range(i + 1, n_current):
                c = colors[idx]
                C[i][j] = c
                C[j][i] = c
                idx += 1
        
        # We need to add vertices n_current to n_target-1
        # We will use a simple recursive backtracking with randomization
        # To avoid deep recursion issues, we'll use a stack or limit depth.
        # Given n_target <= 51, depth is <= 19.
        
        def is_valid_extension(C, new_vertex):
            # Check if C[0:new_vertex] is valid
            # We only need to check triangles involving new_vertex?
            # No, we must ensure the whole graph is valid.
            # But if C[0:new_vertex-1] was valid, and we only added edges for new_vertex,
            # we only need to check triangles involving new_vertex.
            # A triangle (new_vertex, i, j) is monochromatic if:
            # C[new_vertex, i] == C[new_vertex, j] == C[i, j]
            for i in range(new_vertex):
                for j in range(i + 1, new_vertex):
                    if C[new_vertex, i] == C[new_vertex, j] and C[new_vertex, i] == C[i, j]:
                        return False
            return True

        def backtrack(v):
            if time.time() > t_start + time_limit:
                return False
            if v == n_target:
                return True
            
            # Try to assign colors to edges (v, u) for u in 0..v-1
            # Order of u can be randomized
            us = list(range(v))
            random.shuffle(us)
            
            # We assign colors one by one
            # To make it efficient, we can use a constraint propagation or just simple backtracking
            # Simple backtracking:
            # We need to assign a color to each (v, u).
            # Let's try to assign all edges for vertex v.
            
            # There are v edges. 4^v possibilities.
            # For v=19, 4^19 is too large.
            # We need a smarter heuristic.
            
            # Heuristic: For each u, determine forbidden colors.
            # A color c is forbidden for edge (v, u) if there exists w < u such that:
            # C[v, w] == c AND C[u, w] == c
            # This would form a monochromatic triangle (v, u, w).
            
            # We can iterate u in a specific order.
            # Let's try to assign colors to (v, u) for u = v-1 down to 0.
            # Or use a CSP solver approach.
            
            # Simpler: Randomized greedy with restarts.
            # We will try to fill row v.
            
            # For each u, we have a set of allowed colors.
            # If we fix C[v, u1], it restricts C[v, u2].
            
            # Let's use a simple recursive step: pick an unassigned u, try colors.
            # To speed up, pick u with fewest options (MRV heuristic).
            
            assigned = [False] * v
            colors_v = [-1] * v
            
            def solve_step(idx_assigned):
                if time.time() > t_start + time_limit:
                    return False
                if idx_assigned == v:
                    # Check validity
                    if is_valid_extension(C, v):
                        return True
                    return False
                
                # Find unassigned u with fewest allowed colors
                best_u = -1
                best_opts = None
                best_count = 5
                
                for u in range(v):
                    if assigned[u]:
                        continue
                    # Calculate allowed colors for (v, u)
                    allowed = [0, 1, 2, 3]
                    for c in allowed[:]:
                        # Check if c is forbidden
                        # Forbidden if exists w such that C[v, w] == c and C[u, w] == c
                        # Note: C[v, w] might not be assigned yet!
                        # So we can only check constraints involving assigned neighbors.
                        # This makes it a partial check.
                        # If we check against all w, we need C[v, w] to be known.
                        # So this simple MRV doesn't work directly if C[v, w] is unassigned.
                        
                        # Alternative: Just try all 4 colors and check consistency with assigned neighbors.
                        # This is a standard CSP backtrack.
                        pass
                
                # Since dynamic constraint checking is complex, let's just use random order and check consistency with *already assigned* neighbors.
                # This might fail if it creates a conflict later, but backtracking handles it.
                
                # Pick a random unassigned u
                unassigned = [u for u in range(v) if not assigned[u]]
                if not unassigned:
                    return True # Should not happen if idx_assigned < v
                
                u = random.choice(unassigned)
                
                # Try colors in random order
                colors_to_try = [0, 1, 2, 3]
                random.shuffle(colors_to_try)
                
                for c in colors_to_try:
                    # Check if assigning C[v, u] = c is consistent with already assigned C[v, w]
                    consistent = True
                    for w in range(v):
                        if assigned[w] and C[v, w] == c and C[u, w] == c:
                            # Triangle (v, u, w) would be monochromatic c
                            consistent = False
                            break
                    
                    if consistent:
                        assigned[u] = True
                        colors_v[u] = c
                        C[v, u] = c
                        C[u, v] = c
                        
                        if solve_step(idx_assigned + 1):
                            return True
                        
                        # Backtrack
                        assigned[u] = False
                        colors_v[u] = -1
                        C[v, u] = 0
                        C[u, v] = 0
                        
            return solve_step(0)

        # Try to extend
        # We might need multiple restarts if the first path fails
        for attempt in range(50):
            if time.time() > t_start + time_limit:
                break
            # Save state
            C_backup = C.copy()
            if backtrack(n_current):
                # Success
                # Reconstruct colors list
                new_colors = []
                for i in range(n_target):
                    for j in range(i + 1, n_target):
                        new_colors.append(C[i, j])
                return new_colors
            # Restore state
            C[:] = C_backup
            
        return None

    # 1. Get seed 32
    seed_32 = build_seed_32()
    
    # 2. Try to extend to 50
    colors_50 = extend_coloring(seed_32, 32, 50, time_limit=20.0)
    
    if colors_50:
        # 3. Try to extend to 51
        colors_51 = extend_coloring(colors_50, 50, 51, time_limit=20.0)
        if colors_51:
            return {'n': 51, 'colors': colors_51}
        else:
            return {'n': 50, 'colors': colors_50}
    else:
        # Fallback to 32
        return {'n': 32, 'colors': seed_32}
