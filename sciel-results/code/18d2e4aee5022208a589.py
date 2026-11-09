import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def _is_valid_circulant(n, diffs):
        """
        diffs is a list of length floor(n/2).
        diffs[k] is the color for difference k+1.
        Check if no C6 exists.
        """
        # Build full color matrix
        colors = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                diff = j - i
                if diff > n - diff:
                    diff = n - diff
                colors[i][j] = diffs[diff - 1]
                colors[j][i] = colors[i][j]
        
        # Check C6
        # For each color c, check if M_c^4 has any 1s where M_c is 0?
        # No, just check if M_c^4 has any 1s.
        # If M_c^4[u][v] is 1, there is a path of length 4.
        # If we close the loop with u-v, we get C6.
        # But we need to ensure u != v and the path is simple?
        # "Exact search for not necessarily induced cycles".
        # So any closed walk of length 6 is a C6?
        # No, C6 is a cycle of 6 distinct vertices.
        # If M_c^4[u][v] is 1, and (u,v) is an edge in c, does it imply a C6?
        # The path is u-x-y-z-v. If (u,v) is an edge, we have a cycle of length 5? No, 4+1=5 edges.
        # Wait.
        # Cycle: u-x-y-z-v-u. Edges: (u,x), (x,y), (y,z), (z,v), (v,u).
        # Length 5. This is C5.
        # We need C6.
        # Path of length 4: u-x-y-z-v.
        # To make C6, we need another vertex w?
        # No, C6 has 6 vertices.
        # If we have a path of length 5 between u and v, and (u,v) is an edge, it's C6.
        # My previous logic was:
        # If we add edge (w, u) and (w, v) of color c, and there is a path of length 4 between u and v of color c,
        # then w-u-...-v-w is a cycle of length 4 + 2 = 6.
        # So, the condition for validity of the *entire* graph is:
        # For any color c, there is no pair u, v such that:
        # 1. (u, v) is an edge of color c.
        # 2. There is a path of length 4 between u and v in color c.
        
        # So we need to check:
        # For each c:
        #   M_c (adjacency)
        #   M_c^4 (path length 4)
        #   If M_c[u][v] and M_c^4[u][v] are both True, then INVALID.
        
        for c in range(4):
            M = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(i + 1, n):
                    if colors[i][j] == c:
                        M[i][j] = True
                        M[j][i] = True
            
            M2 = M @ M
            M4 = M2 @ M2
            
            # Check overlap
            # If M[i][j] is True and M4[i][j] is True, we have a C6.
            # Note: M4[i][i] might be True (closed walks), but we only care about i != j where M[i][j] is True.
            for i in range(n):
                for j in range(i + 1, n):
                    if M[i][j] and M4[i][j]:
                        return False
        return True

    # Search for circulant colorings
    # Try n = 17, 16, 15
    # n=17: floor(17/2) = 8 variables.
    # 4^8 = 65536 possibilities. Very small!
    # We can brute force n=17 circulant colorings.
    
    for n in [17, 16, 15]:
        if time.time() > deadline - 10:
            break
            
        half_n = n // 2
        # diffs indices 0 to half_n - 1 correspond to differences 1 to half_n
        
        # Use itertools.product
        for combo in product([0, 1, 2, 3], repeat=half_n):
            if time.time() > deadline - 5:
                break
            if _is_valid_circulant(n, combo):
                # Found a valid circulant coloring
                # Construct the full color list
                result_colors = []
                colors = [[0] * n for _ in range(n)]
                for i in range(n):
                    for j in range(i + 1, n):
                        diff = j - i
                        if diff > n - diff:
                            diff = n - diff
                        colors[i][j] = combo[diff - 1]
                for i in range(n):
                    for j in range(i + 1, n):
                        result_colors.append(int(colors[i][j]))
                        
                return {'n': n, 'colors': result_colors}
                
        # If no circulant coloring found for this n, try next n
        # Also, we can try to break the symmetry?
        # But circulant is a good first guess.

    # Fallback: Non-circulant search for n=17
    # If circulant fails, use the backtracking extension method but with more restarts.
    # For now, return best effort.
    
    # Let's try a quick greedy non-circulant for n=17 as fallback
    # (Reusing logic from previous thought)
    
    def _compute_path4(colors, n):
        P = [[[False] * n for _ in range(n)] for _ in range(4)]
        for c in range(4):
            M = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(i + 1, n):
                    if colors[i][j] == c:
                        M[i][j] = True
                        M[j][i] = True
            M2 = M @ M
            M4 = M2 @ M2
            P[c] = [[bool(M4[i][j]) for j in range(n)] for i in range(n)]
        return P

    def _try_extend(n_curr, colors, P, rng):
        if time.time() > deadline - 5:
            return False, None, None
        n_new = n_curr + 1
        assigned = [-1] * n_curr
        
        def backtrack(idx):
            if time.time() > deadline - 2:
                return False
            if idx < 0:
                return True
            v = idx
            color_order = list(range(4))
            rng.shuffle(color_order)
            for c in color_order:
                valid = True
                for u in range(idx + 1, n_curr):
                    if assigned[u] == c:
                        if P[c][v][u]:
                            valid = False
                            break
                if valid:
                    assigned[v] = c
                    if backtrack(idx - 1):
                        return True
                    assigned[v] = -1
            return False

        if backtrack(n_curr - 1):
            new_colors = [row[:] for row in colors]
            for i in range(n_curr):
                new_colors[i].append(assigned[i])
            new_colors.append([0] * n_new)
            for i in range(n_curr):
                new_colors[n_curr][i] = assigned[i]
                new_colors[i][n_curr] = assigned[i]
            new_P = _compute_path4(new_colors, n_new)
            return True, new_colors, new_P
        return False, None, None

    rng = np.random.RandomState(42)
    current_colors = [[0]]
    current_n = 1
    current_P = _compute_path4(current_colors, current_n)
    best_n = 1
    best_colors = current_colors

    for _ in range(50): # 50 restarts
        current_colors = [[0]]
        current_n = 1
        current_P = _compute_path4(current_colors, current_n)
        
        for target_n in range(2, 18):
            if time.time() > deadline - 10:
                break
            ok, new_c, new_P = _try_extend(current_n, current_colors, current_P, rng)
            if ok:
                current_n = target_n
                current_colors = new_c
                current_P = new_P
                if current_n > best_n:
                    best_n = current_n
                    best_colors = current_colors
            else:
                break # Stuck in this restart
        
        if best_n >= 17:
            break

    result_colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            result_colors.append(int(best_colors[i][j]))
    return {'n': best_n, 'colors': result_colors}

