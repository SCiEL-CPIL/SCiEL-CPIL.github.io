import time
import sys

def construct():
    """
    Constructs a 4-coloring of K_n without monochromatic triangles.
    Returns a dict with 'n' and 'colors'.
    """
    # Deadline: 140 seconds to be safe against the 150s kill.
    deadline = time.time() + 140
    
    # --- Step 1: Generate the base K32 coloring (Greenwood-Gleason / Clebsch) ---
    # This is the seed program's logic, which is a valid K32.
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

    base_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                base_colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                base_colors.append(3)
    
    n_current = 32
    colors = base_colors[:]
    
    # Helper to get color of edge (i, j) from the flat list
    # Index for (i, j) with i < j in lexicographic order (0,1), (0,2)...(0,n-1), (1,2)...
    # Count of edges before row i: i*(n-1) - i*(i-1)/2? No, simpler:
    # Edges (0,1)..(0,i-1) is i-1 edges? No.
    # For fixed n, index(i, j) = sum_{k=0}^{i-1} (n-1-k) + (j-1)
    # Let's precompute offsets for the current n.
    
    def get_index(i, j, n):
        # i < j
        # Number of edges with first vertex < i:
        # (0, 1..n-1) -> n-1
        # (1, 2..n-1) -> n-2
        # ...
        # (i-1, i..n-1) -> n-i
        # Sum = (n-1) + (n-2) + ... + (n-i)
        # This is sum_{k=1}^{i} (n-k) = i*n - i*(i+1)/2
        # Then add offset for j in row i: j - i - 1
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    def is_valid_addition(colors_list, n_old, new_vertex_idx):
        # Check if adding new_vertex_idx with the current partial assignments is consistent.
        # Actually, we will check validity incrementally during the search.
        pass

    def search(n_start, colors_list, time_limit):
        """
        Try to extend the coloring from n_start to larger n.
        colors_list contains colors for K_{n_start}.
        Returns (max_n_found, best_colors)
        """
        max_n = n_start
        best_colors = colors_list[:]
        
        # We will try to add vertices one by one.
        # To be efficient, we use backtracking on the assignment of edges for the new vertex.
        
        # State: 
        # n: current number of vertices
        # colors: list of colors for K_n
        
        def backtrack(n, cols):
            nonlocal max_n, best_colors
            if time.time() > time_limit:
                return False
            
            # Update best if we found a larger n
            if n > max_n:
                max_n = n
                best_colors = cols[:]
            
            # Try to add vertex n
            # Edges to add: (0, n), (1, n), ..., (n-1, n)
            # We assign colors for these edges sequentially.
            
            # allowed_colors[i] for edge (i, n)
            # Constraint: For any j < i, triangle (i, j, n) must not be monochromatic.
            # color(i, n) != color(i, j) if color(j, n) == color(i, j)
            
            # We represent the partial assignment for the new vertex as a list of length n.
            # new_vertex_colors[i] = color(i, n)
            
            def assign_new_vertex(idx, current_assignment):
                if time.time() > time_limit:
                    return False
                
                if idx == n:
                    # Successfully assigned all edges for new vertex n
                    # Construct the new full color list
                    new_cols = cols[:]
                    # Append the new edges
                    for i in range(n):
                        new_cols.append(current_assignment[i])
                    
                    # Recurse to next n
                    return backtrack(n + 1, new_cols)
                
                # Determine allowed colors for edge (idx, n)
                allowed = [0, 1, 2, 3]
                for j in range(idx):
                    c_jn = current_assignment[j] # color(j, n)
                    c_ij = cols[get_index(idx, j, n + 1) - (n - idx)] # This index calc is tricky with changing n.
                    # Better to use a 2D matrix or consistent indexing.
                    # Let's switch to a 2D matrix representation internally for speed and clarity, 
                    # then flatten at the end.
                    # But the task requires returning a flat list.
                    # Let's stick to flat list but be careful.
                    # Index of (idx, j) in K_{n+1}? No, (idx, j) is in K_n part.
                    # In the flat list `cols` (size for K_n), the index of (j, idx) where j<idx is:
                    # i = j, j_edge = idx.
                    # idx_in_Kn = j * (2 * n - j - 1) // 2 + (idx - j - 1)
                    
                    c_ij = cols[j * (2 * n - j - 1) // 2 + (idx - j - 1)]
                    
                    if c_ij == c_jn:
                        if c_ij in allowed:
                            allowed.remove(c_ij)
                
                for c in allowed:
                    if assign_new_vertex(idx + 1, current_assignment + [c]):
                        return True
                return False
            
            if assign_new_vertex(0, []):
                return True
            return False

        backtrack(n_start, colors_list)
        return max_n, best_colors

    # Run the search
    # Start from 32
    # Note: The search space grows exponentially. 
    # K32 to K33 might be hard. 
    # However, known results suggest K50 exists. 
    # A simple backtracking might not reach 50 in 140s from 32 without strong heuristics.
    # But even reaching 33 or 34 is an improvement.
    
    # Heuristic: Random restarts or different initial symmetries?
    # The seed coloring is fixed. 
    # We can try to permute the colors of the base K32 to break symmetries?
    # The base K32 is highly symmetric (Clebsch graph). 
    # Let's just run the backtracking. If it's too slow, it will timeout and return 32.
    # To improve chances, we can try a few random permutations of the 4 colors on the base K32.
    
    import random
    random.seed(42)
    
    found_n = 32
    found_colors = colors
    
    # Try a few random color permutations of the base to see if one extends better
    # Actually, permuting colors globally doesn't change the structure, it just relabels.
    # The difficulty is in the extension.
    
    # Let's optimize the backtracking slightly by using a 2D list for the matrix during search.
    
    def search_with_matrix(n_start, colors_list, time_limit):
        nonlocal found_n, found_colors
        
        # Convert flat list to 2D matrix for faster access
        # Matrix M[i][j] for i != j. M[i][j] = M[j][i].
        # Size n_start x n_start.
        
        def build_matrix(n, cols):
            M = [[-1] * n for _ in range(n)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = cols[idx]
                    M[i][j] = c
                    M[j][i] = c
                    idx += 1
            return M

        def flatten_matrix(M, n):
            cols = []
            for i in range(n):
                for j in range(i + 1, n):
                    cols.append(M[i][j])
            return cols

        M = build_matrix(n_start, colors_list)
        
        def backtrack(n, M):
            nonlocal found_n, found_colors
            
            if time.time() > time_limit:
                return False
            
            if n > found_n:
                found_n = n
                found_colors = flatten_matrix(M, n)
            
            # Try to add vertex n
            # Edges (i, n) for i in 0..n-1
            # We store the colors for these in a list `new_edges`
            
            def assign(idx, new_edges):
                if time.time() > time_limit:
                    return False
                
                if idx == n:
                    # Extend M
                    new_M = [row[:] for row in M]
                    for i in range(n):
                        new_M.append([-1] * (n + 1))
                    
                    for i in range(n):
                        c = new_edges[i]
                        new_M[i][n] = c
                        new_M[n][i] = c
                        new_M[n][n] = -1 # Self-loop, irrelevant
                    
                    # Recurse
                    return backtrack(n + 1, new_M)
                
                # Determine allowed colors for (idx, n)
                allowed = [0, 1, 2, 3]
                for j in range(idx):
                    c_jn = new_edges[j]
                    c_ij = M[idx][j]
                    if c_ij == c_jn:
                        if c_ij in allowed:
                            allowed.remove(c_ij)
                
                for c in allowed:
                    if assign(idx + 1, new_edges + [c]):
                        return True
                return False
            
            return assign(0, [])

        backtrack(n_start, M)

    search_with_matrix(32, colors, deadline)
    
    # If we didn't find anything better, return the base
    # The found_n and found_colors are updated by the search.
    # Note: search_with_matrix uses nonlocal variables, so they are updated.
    
    return {'n': found_n, 'colors': found_colors}

# Ensure the function is named construct as per requirements
# The code above defines construct().

if __name__ == '__main__':
    # Small test
    res = construct()
    print(f"Found n={res['n']}")
    # Validate
    n = res['n']
    colors = res['colors']
    # Reconstruct matrix
    M = [[-1]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            M[i][j] = M[j][i] = colors[idx]
            idx += 1
    # Check triangles
    valid = True
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if M[i][j] == M[j][k] == M[i][k]:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    print(f"Valid: {valid}")
