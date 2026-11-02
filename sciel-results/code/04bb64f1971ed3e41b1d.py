<<<<<<< SEARCH
    rank = 55
=======
    rank = 36
>>>>>>> REPLACE
<<<<<<< SEARCH
    # --- Part 2: A2 (2x1) * B2 (1x10) using Schoolbook ---
    # A2 indices: i in {0,1}, j=2 -> U index i*m + 2
    # B2 indices: j=2, k in {0..9} -> V index 2*p + k
    # C2 indices: i in {0,1}, k in {0..9} -> W index k*n + i
    
    # This part is outer product of A_col2 and B_row2.
    # C[i, k] = A[i, 2] * B[2, k]
    
    for i in range(n): # 0, 1
        for k in range(p): # 0..9
            # A index
            u_idx = i * m + 2
            # B index
            v_idx = 2 * p + k
            # C index
            w_idx = k * n + i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
=======
    # --- Part 2: A2 (2x1) * B2 (1x10) using Rank 1 ---
    # A2 indices: i in {0,1}, j=2 -> U index i*m + 2
    # B2 indices: j=2, k in {0..9} -> V index 2*p + k
    # C2 indices: i in {0,1}, k in {0..9} -> W index k*n + i
    
    # This part is outer product of A_col2 and B_row2.
    # C[i, k] = A[i, 2] * B[2, k]
    # Tensor: sum_r U[i,2,r] V[2,k,r] W[k,i,r] = 1
    # Rank 1 is sufficient: U=1, V=1, W=1
    
    for i in range(n):
        U[i * m + 2][col] = 1
        
    for k in range(p):
        V[2 * p + k][col] = 1
        
    for k in range(p):
        for i in range(n):
            W[k * n + i][col] = 1
            
    col += 1
>>>>>>> REPLACE
