"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Split: n=3 -> {0,1}+{2}, m=4 -> {0,1}+{2,3}, p=6 -> {0,1}+{2,3}+{4,5}
    # 2x2x2 blocks use Strassen (rank 7), 1x2x2 blocks use schoolbook (rank 4)
    
    # Strassen decomposition for 2x2x2:
    # C = A*B where A,B are 2x2
    # m1 = (A00+A11)*(B00+B11)
    # m2 = (A10+A11)*B00
    # m3 = A00*(B01-B11)
    # m4 = A11*(B10-B00)
    # m5 = (A00+A01)*B11
    # m6 = (A10-A00)*(B00+B10)
    # m7 = (A01-A11)*(B01+B11)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    def strassen_2x2x2():
        """Return U,V,W for 2x2x2 Strassen. U[4,r], V[4,r], W[4,r] with W[k*2+i,r]"""
        # A indices: A[i,j] -> U[i*2+j, c]
        # B indices: B[j,k] -> V[j*2+k, c]  
        # C indices: C[i,k] -> W[k*2+i, c]
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # m1 = (A00+A11)*(B00+B11)
        U[0][0] = 1; U[3][0] = 1  # A00, A11
        V[0][0] = 1; V[3][0] = 1  # B00, B11
        W[0][0] = 1; W[2][0] = 1  # C00 (k=0,i=0), C10 (k=1,i=0) -> wait
        
        # W[k*2+i, c]: C[i,k] is at W[k*2+i]
        # C00: i=0,k=0 -> W[0*2+0]=W[0]
        # C01: i=0,k=1 -> W[1*2+0]=W[2]
        # C10: i=1,k=0 -> W[0*2+1]=W[1]
        # C11: i=1,k=1 -> W[1*2+1]=W[3]
        
        # m1 = (A00+A11)*(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[2][0] = 1; W[1][0] = 1; W[3][0] = 1  # all four C entries get +1
        
        # m2 = (A10+A11)*B00
        U[2][1] = 1; U[3][1] = 1  # A10, A11
        V[0][1] = 1  # B00
        W[1][1] = 1; W[3][1] = 1  # C10, C11
        
        # m3 = A00*(B01-B11)
        U[0][2] = 1  # A00
        V[1][2] = 1; V[3][2] = -1  # B01, -B11
        W[2][2] = 1; W[3][2] = 1  # C01, C11
        
        # m4 = A11*(B10-B00)
        U[3][3] = 1  # A11
        V[2][3] = 1; V[0][3] = -1  # B10, -B00
        W[0][3] = 1; W[1][3] = 1  # C00, C10
        
        # m5 = (A00+A01)*B11
        U[0][4] = 1; U[1][4] = 1  # A00, A01
        V[3][4] = 1  # B11
        W[0][4] = -1; W[2][4] = 1  # -C00, +C01
        
        # m6 = (A10-A00)*(B00+B10)
        U[2][5] = 1; U[0][5] = -1  # A10, -A00
        V[0][5] = 1; V[2][5] = 1  # B00, B10
        W[3][5] = 1  # C11
        
        # m7 = (A01-A11)*(B01+B11)
        U[1][6] = 1; U[3][6] = -1  # A01, -A11
        V[1][6] = 1; V[3][6] = 1  # B01, B11
        W[0][6] = 1  # C00
        
        return U, V, W
    
    def schoolbook_1x2x2():
        """Return U,V,W for 1x2x2 schoolbook. U[2,r], V[2,r], W[2,r] with W[k*1+i,r]=W[k,r]"""
        # A is 1x2: A[0,j] -> U[j, c]
        # B is 2x2: B[j,k] -> V[j*2+k, c]
        # C is 1x2: C[0,k] -> W[k*1+0] = W[k, c]
        U = [[0]*4 for _ in range(2)]
        V = [[0]*4 for _ in range(4)]
        W = [[0]*4 for _ in range(2)]
        
        for j in range(2):
            for k in range(2):
                c = j*2+k
                U[j][c] = 1
                V[j*2+k][c] = 1
                W[k][c] = 1
        
        return U, V, W
    
    # Build full decomposition
    # n=3, m=4, p=6
    # U shape: (n*m, rank) = (12, rank)
    # V shape: (m*p, rank) = (24, rank)
    # W shape: (p*n, rank) = (18, rank)
    
    # We'll build lists of (local_U, local_V, local_W, i_offset, j_offset, k_offset)
    # and then assemble
    
    U_rows = n * m  # 12
    V_rows = m * p  # 24
    W_rows = p * n  # 18
    
    # Collect all terms
    terms = []  # each term: (U_local, V_local, W_local, i_off, j_off, k_off)
    
    # i blocks: [0,1] (size 2), [2] (size 1)
    # j blocks: [0,1] (size 2), [2,3] (size 2)
    # k blocks: [0,1] (size 2), [2,3] (size 2), [4,5] (size 2)
    
    i_blocks = [(0, 2), (2, 1)]  # (start, size)
    j_blocks = [(0, 2), (2, 2)]
    k_blocks = [(0, 2), (2, 2), (4, 2)]
    
    for i_start, i_size in i_blocks:
        for j_start, j_size in j_blocks:
            for k_start, k_size in k_blocks:
                if i_size == 2 and j_size == 2 and k_size == 2:
                    U_loc, V_loc, W_loc = strassen_2x2x2()
                else:
                    # 1x2x2
                    U_loc, V_loc, W_loc = schoolbook_1x2x2()
                terms.append((U_loc, V_loc, W_loc, i_start, j_start, k_start))
    
    # Calculate total rank
    total_rank = sum(len(U_loc[0]) for U_loc, _, _, _, _, _ in terms)
    
    # Assemble
    U = [[0] * total_rank for _ in range(U_rows)]
    V = [[0] * total_rank for _ in range(V_rows)]
    W = [[0] * total_rank for _ in range(W_rows)]
    
    col = 0
    for U_loc, V_loc, W_loc, i_start, j_start, k_start in terms:
        r = len(U_loc[0])
        for c in range(r):
            # U: row = i*m + j, local row = li * j_size + lj
            for li in range(len(U_loc)):
                i_global = i_start + li
                j_local = li % j_size  # This isn't right for 2x2x2
                # Actually for U in 2x2x2, local index is i_local * 2 + j_local
                # But we need to map to global: i_global = i_start + i_local, j_global = j_start + j_local
                # For 2x2x2: U_loc[i_local*2 + j_local, c]
                # For 1x2x2: U_loc[j_local, c] where i_local=0 always
                
                if len(U_loc) == 4:  # 2x2x2
                    i_local = li // 2
                    j_local = li % 2
                else:  # 1x2x2
                    i_local = 0
                    j_local = li
                
                i_global = i_start + i_local
                j_global = j_start + j_local
                U[i_global * m + j_global][col + c] = U_loc[li][c]
            
            # V: row = j*p + k, local row = lj * k_size + lk
            for lj in range(len(V_loc)):
                if len(V_loc) == 4:  # 2x2x2
                    j_local = lj // 2
                    k_local = lj % 2
                else:  # 1x2x2: V is 4x4, row = j_local*2 + k_local
                    j_local = lj // 2
                    k_local = lj % 2
                
                j_global = j_start + j_local
                k_global = k_start + k_local
                V[j_global * p + k_global][col + c] = V_loc[lj][c]
            
            # W: row = k*n + i, local row = lk * i_size + li
            for lk in range(len(W_loc)):
                if len(W_loc) == 4:  # 2x2x2: W local = k_local*2 + i_local
                    k_local = lk // 2
                    i_local = lk % 2
                else:  # 1x2x2: W is 2x4, row = k_local (since i_size=1)
                    k_local = lk
                    i_local = 0
                
                k_global = k_start + k_local
                i_global = i_start + i_local
                W[k_global * n + i_global][col + c] = W_loc[lk][c]
        
        col += r
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
