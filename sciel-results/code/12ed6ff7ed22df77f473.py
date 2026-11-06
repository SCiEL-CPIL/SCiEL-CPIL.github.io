"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Greedy randomized construction for R(3,3,3,3) > n.
    We try to build a 4-coloring of K_n with no monochromatic triangle.
    
    Strategy:
    - Start with a known good base (e.g., n=16 or n=32 from seed logic, or start small)
    - Iteratively add vertices, choosing the color assignment that maximizes minimum slack
    - Use random restarts with perturbations if stuck
    - Target: maximize n (Chung's record is 50, so we aim for at least 32, hopefully more)
    
    For efficiency within 120s, we use a structured greedy approach with backtracking limited.
    """
    import time
    start_time = time.time()
    TIMEOUT = 110  # leave buffer
    
    def time_left():
        return TIMEOUT - (time.time() - start_time)
    
    # Precompute: for a given coloring of K_n (stored as dict or list), 
    # we need to check if adding vertex v with a specific color assignment to all existing vertices
    # creates a monochromatic triangle.
    
    # Representation: colors is a list of length n*(n-1)//2, indexed by pair (i,j) in lexicographic order.
    # Let's use a 2D list for internal representation for clarity, then convert.
    
    def pair_index(n, i, j):
        """Index in the flat list for pair (i,j) with i<j in K_n."""
        # Number of pairs before row i: i*(i-1)//2 + (i-1) ... let's derive properly
        # Pairs: (0,1),(0,2),...,(0,n-1),(1,2),(1,3),...
        # For vertex i, pairs with j>i: indices start at sum_{k=0}^{i-1} (n-1-k)
        # Actually, for fixed n, the index of (i,j) where i<j:
        # idx = i*(n-1) - i*(i+1)//2 + (j - i - 1)
        # Let's verify: n=4
        # (0,1): 0*3 - 0 + 0 = 0 ✓
        # (0,2): 0*3 - 0 + 1 = 1 ✓
        # (0,3): 0*3 - 0 + 2 = 2 ✓
        # (1,2): 1*3 - 1 + 0 = 2 ✓
        # (1,3): 1*3 - 1 + 1 = 3 ✓
        # (2,3): 2*3 - 3 + 0 = 3 ✓
        # Wait, (1,2) should be index 3, (1,3) index 4, (2,3) index 5 for n=4.
        # Total pairs = 6.
        # Let me recount: (0,1)=0, (0,2)=1, (0,3)=2, (1,2)=3, (1,3)=4, (2,3)=5
        # Formula: idx = i*(n-1) - i*(i+1)//2 + (j-i-1)
        # (1,2): 1*3 - 1*2//2 + (2-1-1) = 3 - 1 + 0 = 2. Wrong, should be 3.
        # Let's use a different approach. Precompute indices.
        
        pass
    
    # Better: just use a 2D array internally
    def make_color_matrix(n):
        return [[-1]*n for _ in range(n)]
    
    def matrix_to_list(n, mat):
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(mat[i][j])
        return colors
    
    def check_no_mono_triangle(n, mat):
        """Check if the coloring has no monochromatic triangle."""
        for c in range(4):
            # Build adjacency for color c
            adj = [[False]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if mat[i][j] == c:
                        adj[i][j] = True
                        adj[j][i] = True
            # Check for triangle
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        for k in range(j+1, n):
                            if adj[i][k] and adj[j][k]:
                                return False
        return True
    
    def count_conflicts(n, mat, v_new, color_assignment):
        """
        For a new vertex v_new (index n), given color_assignment (list of length n, 
        where color_assignment[i] is the color of edge (i, v_new)),
        count how many monochromatic triangles would be created.
        """
        conflicts = 0
        for c in range(4):
            # Find all vertices i where color_assignment[i] == c
            verts_c = [i for i in range(n) if color_assignment[i] == c]
            # For each pair (i,j) in verts_c where mat[i][j] == c, we have a conflict
            for idx_i in range(len(verts_c)):
                i = verts_c[idx_i]
                for idx_j in range(idx_i+1, len(verts_c)):
                    j = verts_c[idx_j]
                    if mat[i][j] == c:
                        conflicts += 1
        return conflicts
    
    def greedy_add_vertex(n, mat, max_iters=10000):
        """
        Try to add a new vertex to the coloring of K_n.
        Returns (new_n, new_mat) if successful, else None.
        """
        best_score = -1
        best_assignment = None
        
        # We'll use a heuristic: try random assignments and keep the one with fewest conflicts
        # But to be smarter, we can try to find an assignment with 0 conflicts
        
        # First, let's try a few random assignments
        for attempt in range(50):
            if time_left() < 5:
                return None
            
            # Random assignment
            assignment = [random.randint(0, 3) for _ in range(n)]
            conflicts = count_conflicts(n, mat, n, assignment)
            
            if conflicts == 0:
                # Success!
                new_mat = [row[:] for row in mat]
                for i in range(n):
                    new_mat.append([-1]*n)
                new_n = n + 1
                new_mat = [row[:] for row in new_mat]
                # Actually, let's just create a new matrix
                new_mat = [[-1]*(n+1) for _ in range(n+1)]
                for i in range(n):
                    for j in range(i+1, n):
                        new_mat[i][j] = mat[i][j]
                        new_mat[j][i] = mat[i][j]
                for i in range(n):
                    new_mat[i][n] = assignment[i]
                    new_mat[n][i] = assignment[i]
                return (new_n, new_mat)
            
            if conflicts < best_score or best_assignment is None:
                best_score = conflicts
                best_assignment = assignment
        
        # If no zero-conflict assignment found, try a more systematic approach
        # For each vertex i, the color of (i, new) is constrained by the colors of edges incident to i
        # Specifically, for each color c, if there's a pair (j,k) with mat[j][k]=c and both 
        # color_assignment[j]=c and color_assignment[k]=c, that's a conflict.
        
        # Let's try a local search: start with random, then flip colors to reduce conflicts
        assignment = best_assignment[:]
        for _ in range(200):
            if time_left() < 5:
                return None
            conflicts = count_conflicts(n, mat, n, assignment)
            if conflicts == 0:
                new_mat = [[-1]*(n+1) for _ in range(n+1)]
                for i in range(n):
                    for j in range(i+1, n):
                        new_mat[i][j] = mat[i][j]
                        new_mat[j][i] = mat[i][j]
                for i in range(n):
                    new_mat[i][n] = assignment[i]
                    new_mat[n][i] = assignment[i]
                return (new_n, new_mat)
            
            # Try flipping one vertex's color
            best_flip = None
            best_conflicts = conflicts
            for i in range(n):
                old_color = assignment[i]
                for new_color in range(4):
                    if new_color == old_color:
                        continue
                    assignment[i] = new_color
                    new_conflicts = count_conflicts(n, mat, n, assignment)
                    assignment[i] = old_color
                    if new_conflicts < best_conflicts:
                        best_conflicts = new_conflicts
                        best_flip = (i, new_color)
            
            if best_flip is None:
                # No improvement, restart
                assignment = [random.randint(0, 3) for _ in range(n)]
            else:
                assignment[best_flip[0]] = best_flip[1]
        
        return None
    
    # Start with a small valid coloring
    # K_1 is trivially valid
    n = 1
    mat = [[-1]*1 for _ in range(1)]
    
    # Try to grow
    max_n = 0
    best_result = None
    
    # First, let's try to build up from scratch with multiple restarts
    for restart in range(20):
        if time_left() < 10:
            break
        
        # Start fresh
        n = 1
        mat = [[-1]*1 for _ in range(1)]
        
        # Grow
        while time_left() > 10:
            result = greedy_add_vertex(n, mat)
            if result is None:
                break
            n, mat = result
            if n > max_n:
                max_n = n
                best_result = (n, [row[:] for row in mat])
        
        # If we got stuck, we could try perturbation, but for simplicity, just restart
    
    # Use the best result
    if best_result is not None:
        n, mat = best_result
        colors = matrix_to_list(n, mat)
        return {'n': n, 'colors': colors}
    else:
        # Fallback: return n=1
        return {'n': 1, 'colors': []}
# EVOLVE-BLOCK-END
