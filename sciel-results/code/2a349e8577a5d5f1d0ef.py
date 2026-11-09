def find_coloring(n, time_limit):
    # Initialize random coloring
    colors = np.random.randint(0, 4, size=(n, n))
    # Symmetrize and zero diagonal
    colors = (colors + colors.T) // 2
    np.fill_diagonal(colors, -1) # -1 means no edge
    
    # Adjacency matrices for each color
    A = [np.zeros((n, n), dtype=int) for _ in range(4)]
    for i in range(n):
        for j in range(i+1, n):
            c = colors[i, j]
            A[c][i, j] = 1
            A[c][j, i] = 1
            
    # Calculate initial cost
    total_cost = 0
    for c in range(4):
        A4 = A[c] @ A[c] @ A[c] @ A[c]
        # Sum of all entries in A4 is 2 * number of C6s?
        # No, A4[u,v] is number of paths of length 4 from u to v.
        # Each C6 contributes to 6 such paths?
        # A C6 is u-v-w-x-y-z-u.
        # Paths of length 4 in the cycle:
        # u to w (2 ways? no, simple path)
        # In a simple cycle C6, for any pair of vertices at distance 4, there is 1 path of length 4.
        # Number of pairs at distance 4 in C6 is 6.
        # So sum(A4) = 6 * (# of C6s).
        # We just need total_cost > 0 check.
        total_cost += np.sum(A4)
        
    if total_cost == 0:
        return extract_colors(colors)

    deadline = time.time() + time_limit
    
    while time.time() < deadline:
        # Pick random edge
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v: continue
        
        c_old = colors[u, v]
        
        # Calculate k_old: paths of length 4 in A[c_old]
        # Note: A[c_old] contains edge (u,v).
        # Does (A[c_old]^4)[u,v] count paths using (u,v)?
        # Path length 4 cannot use edge (u,v) directly as a single step, but can it traverse it?
        # A path is a sequence of vertices.
        # If the path is u-v-...-v, it repeats v.
        # We are counting walks, not simple paths.
        # This is a problem.
        # My previous analysis assumed simple paths or that walks don't create false positives for C6 existence?
        # If there is a walk of length 4, it implies a cycle?
        # Not necessarily a simple C6.
        # However, if there is a simple C6, there is a walk.
        # If there is a walk but no simple C6, we might think there is a C6 (false positive).
        # This makes the cost function noisy (overestimates).
        # But if we minimize the number of walks, we will also minimize the number of simple cycles?
        # Likely yes.
        # Let's proceed with walks. It's much faster.
        
        A4_old = A[c_old] @ A[c_old] @ A[c_old] @ A[c_old]
        k_old = A4_old[u, v]
        
        best_delta = 0
        best_c_new = c_old
        
        for c_new in range(4):
            if c_new == c_old: continue
            
            A4_new = A[c_new] @ A[c_new] @ A[c_new] @ A[c_new]
            k_new = A4_new[u, v]
            
            delta = k_new - k_old
            
            if delta < best_delta:
                best_delta = delta
                best_c_new = c_new
                
        if best_c_new != c_old:
            # Flip
            colors[u, v] = best_c_new
            colors[v, u] = best_c_new
            
            A[c_old][u, v] = 0
            A[c_old][v, u] = 0
            A[best_c_new][u, v] = 1
            A[best_c_new][v, u] = 1
            
            total_cost += best_delta
            
            if total_cost == 0:
                return extract_colors(colors)
                
    return None # Failed to find valid
