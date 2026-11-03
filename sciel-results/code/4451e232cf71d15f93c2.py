import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # We will construct a rank-98 decomposition using two rank-49 Strassen decompositions for 4x4x4
    
    # Step 1: Generate rank-49 decomposition for 4x4x4 using recursive Strassen
    
    def get_strassen_2x2():
        # Returns U, V, W for 2x2x2, rank 7
        # U shape (4, 7), V shape (4, 7), W shape (4, 7)
        # Indices:
        # A: 00=0, 01=1, 10=2, 11=3
        # B: 00=4, 01=5, 10=6, 11=7
        # C: 00=8, 01=9, 10=10, 11=11
        # But the interface expects U (n*m, rank), V (m*p, rank), W (p*n, rank)
        # For 2x2x2: n=2, m=2, p=2.
        # U rows: i*m+j. 00=0, 01=1, 10=2, 11=3.
        # V rows: j*p+k. 00=0, 01=1, 10=2, 11=3.
        # W rows: k*n+i. 00=0, 01=1, 10=2, 11=3.
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Term 0: (A00+A11)(B00+B11) -> C00, C11
        # U: A00(0), A11(3)
        # V: B00(0), B11(3)
        # W: C00(0), C11(3)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # Term 1: (A10+A11)B00 -> C10(+), C11(-)
        # U: A10(2), A11(3)
        # V: B00(0)
        # W: C10(2), C11(3)
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # Term 2: A00(B01-B11) -> C01(+), C11(+)
        # U: A00(0)
        # V: B01(1), B11(3)
        # W: C01(1), C11(3)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # Term 3: A11(B10-B00) -> C00(+), C10(+)
        # U: A11(3)
        # V: B10(2), B00(0)
        # W: C00(0), C10(2)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # Term 4: (A00+A01)B11 -> C00(-), C01(+)
        # U: A00(0), A01(1)
        # V: B11(3)
        # W: C00(0), C01(1)
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # Term 5: (A10-A00)(B00+B01) -> C11(+)
        # U: A10(2), A00(0)
        # V: B00(0), B01(1)
        # W: C11(3)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # Term 6: (A01-A11)(B10+B11) -> C00(+)
        # U: A01(1), A11(3)
        # V: B10(2), B11(3)
        # W: C00(0)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def expand_strassen(N, U_sub, V_sub, W_sub):
        # Expands a decomposition of size N/2 to size N
        # U_sub shape: (N/2 * N/2, R_sub)
        # Returns U, V, W of shape (N*N, 7*R_sub)
        
        half = N // 2
        R_sub = U_sub.shape[1]
        R_new = 7 * R_sub
        
        U = np.zeros((N*N, R_new))
        V = np.zeros((N*N, R_new))
        W = np.zeros((N*N, R_new))
        
        # Block mappings for the 7 terms
        # Each term has:
        # u_blocks: list of (block_idx, coeff)
        # v_blocks: list of (block_idx, coeff)
        # w_blocks: list of (block_idx, coeff)
        
        terms = [
            # Term 0
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            # Term 1
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            # Term 2
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            # Term 3
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            # Term 4
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            # Term 5
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            # Term 6
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
        ]
        
        for t in range(7):
            u_blocks, v_blocks, w_blocks = terms[t]
            col_offset = t * R_sub
            
            for c_sub in range(R_sub):
                c_new = col_offset + c_sub
                
                # Process U
                for i_sub in range(half * half):
                    val = U_sub[i_sub, c_sub]
                    if val != 0:
                        i_local, j_local = divmod(i_sub, half)
                        for b_idx, coeff in u_blocks:
                            r_block, c_block = divmod(b_idx, 2)
                            i_global = r_block * half + i_local
                            j_global = c_block * half + j_local
                            idx_global = i_global * N + j_global
                            U[idx_global, c_new] += val * coeff
                            
                # Process V
                for j_sub in range(half * half):
                    val = V_sub[j_sub, c_sub]
                    if val != 0:
                        j_local, k_local = divmod(j_sub, half)
                        for b_idx, coeff in v_blocks:
                            r_block, c_block = divmod(b_idx, 2)
                            j_global = r_block * half + j_local
                            k_global = c_block * half + k_local
                            idx_global = j_global * N + k_global
                            V[idx_global, c_new] += val * coeff
                            
                # Process W
                for k_sub in range(half * half):
                    val = W_sub[k_sub, c_sub]
                    if val != 0:
                        k_local, i_local = divmod(k_sub, half)
                        for b_idx, coeff in w_blocks:
                            r_block, c_block = divmod(b_idx, 2)
                            k_global = r_block * half + k_local
                            i_global = c_block * half + i_local
                            idx_global = k_global * N + i_global
                            W[idx_global, c_new] += val * coeff
                            
        return U, V, W

    # Generate 4x4x4 decomposition
    U2, V2, W2 = get_strassen_2x2()
    U4, V4, W4 = expand_strassen(4, U2, V2, W2)
    
    rank_444 = U4.shape[1] # Should be 49
    rank_total = 2 * rank_444 # 98
    
    # Initialize global matrices
    U = np.zeros((n*m, rank_total))
    V = np.zeros((m*p, rank_total))
    W = np.zeros((p*n, rank_total))
    
    # Fill first half (cols 0-3 of B, 0-3 of C)
    for c in range(rank_444):
        U[:, c] = U4[:, c]
        
        # Remap V4
        # V4 row: j*4 + k (k in 0..3)
        # Global V row: j*8 + k
        for r_sub in range(m*m):
            val = V4[r_sub, c]
            if val != 0:
                j, k = divmod(r_sub, m) # m=4
                r_global = j * p + k # p=8
                V[r_global, c] = val
                
        # Remap W4
        # W4 row: k*4 + i (k in 0..3)
        # Global W row: k*4 + i
        for r_sub in range(p*n): # Wait, W4 is for 4x4x4, so p_sub=4, n_sub=4
             # W4 shape is (4*4, 49) = (16, 49)
             # Row index: k_sub * 4 + i_sub
             pass
        
        # Let's do W4 remapping properly
        # W4 row index: k_sub * n_sub + i_sub, where n_sub=4
        # Global W row index: k_global * n + i, where n=4
        # For first half, k_global = k_sub
        for r_sub in range(4*4):
            val = W4[r_sub, c]
            if val != 0:
                k_sub, i_sub = divmod(r_sub, 4)
                k_global = k_sub
                i_global = i_sub
                r_global = k_global * n + i_global
                W[r_global, c] = val

    # Fill second half (cols 4-7 of B, 4-7 of C)
    for c in range(rank_444):
        c_global = rank_444 + c
        U[:, c_global] = U4[:, c]
        
        # Remap V4
        # V4 row: j*4 + k (k in 0..3)
        # Global V row: j*8 + (k+4)
        for r_sub in range(m*m):
            val = V4[r_sub, c]
            if val != 0:
                j, k = divmod(r_sub, m)
                r_global = j * p + (k + 4)
                V[r_global, c_global] = val
                
        # Remap W4
        # W4 row: k*4 + i (k in 0..3)
        # Global W row: (k+4)*4 + i
        for r_sub in range(4*4):
            val = W4[r_sub, c]
            if val != 0:
                k_sub, i_sub = divmod(r_sub, 4)
                k_global = k_sub + 4
                i_global = i_sub
                r_global = k_global * n + i_global
                W[r_global, c_global] = val

    # Convert to lists of lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
