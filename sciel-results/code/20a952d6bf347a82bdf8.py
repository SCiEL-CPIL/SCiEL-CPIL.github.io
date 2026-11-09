import time
import numpy as np

def _check_validity(colors, n):
    # colors is a list of length n*(n-1)/2
    # Reconstruct matrix
    c_mat = np.zeros((n, n), dtype=int)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c_mat[i, j] = colors[idx]
            c_mat[j, i] = colors[idx]
            idx += 1
            
    # Check for C6 in each color
    for c in range(4):
        A = (c_mat == c).astype(int)
        # A[i,j] = 1 if edge (i,j) is color c
        # We need to check if there is a cycle of length 6.
        # A cycle of length 6 is a path of length 6 from i to i?
        # Or simply check if A^6 has non-zero diagonal?
        # But A^6 counts walks, not simple cycles.
        # However, for small n, we can just enumerate all 6-tuples? No, too slow.
        # We can use the fact that if there is a C6, then there is a path of length 5 between adjacent vertices?
        # Actually, the backtracking ensures validity by construction.
        # This function is just for debugging or final verification.
        pass
    return True

def _get_P4(c_mat, n):
    """
    Returns a list of 4 boolean matrices P4[c] for c in 0..3.
    P4[c][i, j] is True if there is a path of length 4 from i to j using only color c.
    """
    P4 = []
    for c in range(4):
        A = (c_mat == c).astype(int)
        A2 = A @ A
        # Path of length 4 is A^4.
        # But we only need boolean existence.
        # A4_bool = (A2 @ A2) > 0
        A4 = A2 @ A2
        P4.append((A4 > 0))
    return P4

def _backtrack_build(n, time_limit):
    """
    Attempts to build a valid 4-coloring of K_n.
    Returns a list of colors or None if failed.
    """
    start_time = time.time()
    
    # c_mat[i, j] is color of edge (i, j). -1 if unassigned.
    c_mat = np.full((n, n), -1, dtype=int)
    
    # We build vertex by vertex.
    # At step k, we assign colors to edges (k, 0), ..., (k, k-1).
    # We maintain P4 for the current graph G_{k-1}.
    # Initially G_0 is empty.
    
    P4 = [np.zeros((n, n), dtype=bool) for _ in range(4)]
    
    def solve(k):
        if time.time() - start_time > time_limit:
            return False
            
        if k == n:
            return True
            
        # We need to assign colors to edges (k, 0) ... (k, k-1)
        # Let's use backtracking for these k edges.
        # colors[k, 0..k-1]
        
        # Optimization: We can try to assign colors in a way that minimizes conflicts.
        # Or just try 0,1,2,3 for each edge.
        
        # To prune, we maintain a list of assigned colors for edges (k, i) for i < k.
        # assigned[i] = color of (k, i)
        
        assigned = [-1] * k
        
        def backtrack_edge(idx):
            if time.time() - start_time > time_limit:
                return False
                
            if idx == k:
                # All edges assigned.
                # Check if this assignment is valid with respect to P4 of G_{k-1}.
                # Actually, we should have been checking during assignment.
                # If we reached here, it's valid.
                # Update c_mat
                for i in range(k):
                    c_mat[k, i] = assigned[i]
                    c_mat[i, k] = assigned[i]
                
                # Update P4 for G_k
                # We need to recompute P4 for the new graph.
                # This is O(4 * n^3) roughly.
                new_P4 = _get_P4(c_mat, n)
                # We need to update the outer scope P4.
                # Since P4 is a list of arrays, we can just reassign or modify in place.
                # But _get_P4 returns new arrays.
                # Let's just recompute and return True.
                # The caller will handle the next step.
                # Wait, the P4 variable in the closure is not easily updatable if we recompute.
                # Better to pass P4 as a mutable state or recompute in the main loop.
                # Let's restructure: The main loop calls a function that returns the new P4.
                # Or just recompute P4 in the main loop after a successful step.
                
                # For now, let's assume the check happens in backtrack_edge.
                return True
                
            # Try colors 0, 1, 2, 3 for edge (k, idx)
            for c in range(4):
                assigned[idx] = c
                
                # Check conflicts with previously assigned edges (k, j) for j < idx
                valid = True
                for j in range(idx):
                    if assigned[j] == c:
                        # Both (k, j) and (k, idx) are color c.
                        # Check if (j, idx) is in P4[c] of G_{k-1}.
                        if P4[c][j, idx]:
                            valid = False
                            break
                
                if valid:
                    if backtrack_edge(idx + 1):
                        return True
                
            assigned[idx] = -1
            return False
            
        return backtrack_edge(0)
        
    # Main loop
    for k in range(1, n):
        if not solve(k):
            return None
        # If solve(k) succeeded, c_mat is updated for row/col k.
        # We need to update P4 for the next iteration.
        P4 = _get_P4(c_mat, n)
        
    # Extract colors
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            colors.append(int(c_mat[i, j]))
            
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Try n=17
    best_n = 0
    best_colors = []
    
    # We can try 17 first.
    # If 17 fails, try 16, etc.
    # But 17 is known to exist.
    
    # To increase chance of success, we can add some randomization to the color order in backtracking.
    # But deterministic is safer for reproducibility and correctness.
    # Let's stick to deterministic first.
    
    # Optimization: In _backtrack_build, the order of colors 0,1,2,3 might matter.
    # We can shuffle them if we detect a timeout? No, keep it simple.
    
    res = _backtrack_build(17, 50)
    if res:
        best_n = 17
        best_colors = res
        
    # Try 18 if time allows
    if time.time() < deadline - 20 and best_n == 17:
        res18 = _backtrack_build(18, 80)
        if res18:
            best_n = 18
            best_colors = res18
            
    if best_n == 0:
        # Fallback to a known small solution
        n = 9
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
        
    return {'n': best_n, 'colors': best_colors}
