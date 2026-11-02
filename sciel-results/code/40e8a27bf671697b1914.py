import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Build the multiplication tensor T[i,j,k] where:
    # i indexes A (n*m = 8 elements), j indexes B (m*p = 28 elements), k indexes C (p*n = 14 elements)
    # T[i,j,k] = 1 if the (i,j) pair in A and (j,k) pair in B contribute to C[k]
    # More precisely: A has shape (n,m), B has shape (m,p), C has shape (n,p)
    # C[a,b] = sum_j A[a,j] * B[j,b]
    # So the tensor T[A_idx, B_idx, C_idx] = 1 if A_idx=(a,j), B_idx=(j,b), C_idx=(b,a)
    # Using trace convention: C_idx = k*n + i where k=b, i=a
    
    # Flatten indices:
    # A_idx = a*m + j, where a in [0,n), j in [0,m)
    # B_idx = j*p + b, where j in [0,m), b in [0,p)
    # C_idx = b*n + a, where b in [0,p), a in [0,n)
    
    A_dim = n * m  # 8
    B_dim = m * p  # 28
    C_dim = p * n  # 14
    
    # Build tensor T of shape (A_dim, B_dim, C_dim)
    T = np.zeros((A_dim, B_dim, C_dim), dtype=float)
    
    for a in range(n):
        for j in range(m):
            for b in range(p):
                A_idx = a * m + j
                B_idx = j * p + b
                C_idx = b * n + a
                T[A_idx, B_idx, C_idx] = 1.0
    
    # MPS approach: reshape T as a matrix and do SVD
    # We can reshape T to (A_dim, B_dim * C_dim) and do SVD
    # Or we can try different groupings to find low-rank structure
    
    # Strategy: Try to find a low-rank decomposition by SVD on different reshaping
    # T[a,j, b,c] where we group (a,j) as A, (b,c) as... wait, C is (b,a) not (b,c)
    
    # Let me think differently. The tensor T[i,j,k] where:
    # i = a*m + j (A index)
    # j = j'*p + b (B index)  
    # k = b*n + a (C index)
    # The constraint is that the j in A_idx must equal the j' in B_idx
    
    # So T[i, j, k] is non-zero only when:
    # i = a*m + j_a, j = j_b*p + b, k = b*n + a, with j_a = j_b
    
    # This means T has a very specific sparse structure.
    # Let's try SVD on T reshaped as (A_dim, B_dim*C_dim)
    
    T_mat1 = T.reshape(A_dim, B_dim * C_dim)
    U1, S1, Vt1 = np.linalg.svd(T_mat1, full_matrices=False)
    
    # Also try (A_dim*B_dim, C_dim)
    T_mat2 = T.reshape(A_dim * B_dim, C_dim)
    U2, S2, Vt2 = np.linalg.svd(T_mat2, full_matrices=False)
    
    # And (A_dim*C_dim, B_dim)
    T_mat3 = T.reshape(A_dim * C_dim, B_dim)
    U3, S3, Vt3 = np.linalg.svd(T_mat3, full_matrices=False)
    
    # The rank of the tensor is at least the max of these matrix ranks
    # But we want the exact tensor rank (sum of rank-1 tensors)
    
    # For now, let's use the schoolbook decomposition as fallback
    # but check if SVD gives us a lower rank
    rank1 = len(S1[S1 > 1e-10])
    rank2 = len(S2[S2 > 1e-10])
    rank3 = len(S3[S3 > 1e-10])
    
    # The tensor rank is at least max(rank1, rank2, rank3)
    # But the actual tensor rank could be higher
    
    # For this specific problem, let's just use the known optimal or near-optimal
    # The schoolbook gives 56. Can we do better?
    
    # Actually, for 2x4x7, the schoolbook rank is 2*4*7 = 56
    # Using Strassen-style block decomposition might help
    
    # Let me try a block approach:
    # Split A into 2x2 blocks (2 blocks along m dimension)
    # Split B into 2x2 blocks
    # Use Strassen for 2x2x2 (rank 7) and 2x2x3 (rank 12 schoolbook)
    
    # A is 2x4: two 2x2 blocks
    # B is 4x7: two 2x7 blocks
    # C is 2x7: two 2x7 blocks
    
    # C_left = A_left * B_left + A_right * B_right (if we split m in half)
    # Wait, A is (2,4), B is (4,7)
    # Split A = [A1 | A2] where A1, A2 are (2,2)
    # Split B = [B1; B2] where B1, B2 are (2,7)
    # C = A1*B1 + A2*B2
    
    # A1*B1: 2x2 * 2x7 = 2x7
    # A2*B2: 2x2 * 2x7 = 2x7
    
    # For 2x2 * 2x7:
    # Split 2x7 into 2x2 + 2x2 + 2x3 (three blocks)
    # A1*B1 = A1*[B1a B1b B1c] = [A1*B1a, A1*B1b, A1*B1c]
    # A1*B1a: 2x2*2x2 rank 7 (Strassen)
    # A1*B1b: 2x2*2x2 rank 7 (Strassen)  
    # A1*B1c: 2x2*2x3 rank 12 (schoolbook)
    # Total for A1*B1: 7+7+12 = 26
    
    # Similarly A2*B2: 26
    # Total: 52
    
    # That's better than 56! Let me implement this.
    
    rank = 52
    U = [[0] * rank for _ in range(A_dim)]
    V = [[0] * rank for _ in range(B_dim)]
    W = [[0] * rank for _ in range(C_dim)]
    
    col = 0
    
    # Helper: Strassen 2x2x2 decomposition
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B
    # 7 products:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def strassen_222(U_mat, V_mat, W_mat, col_start, a_base, b_base, c_base):
        """
        Add Strassen 2x2x2 terms starting at col_start.
        a_base: offset for A indices (2x2 block)
        b_base: offset for B indices (2x2 block)  
        c_base: offset for C indices (2x2 block)
        Returns col_start + 7
        """
        # A indices: a00=a_base, a01=a_base+1, a10=a_base+2, a11=a_base+3
        # B indices: b00=b_base, b01=b_base+1, b10=b_base+2, b11=b_base+3
        # C indices: c00=c_base, c01=c_base+1, c10=c_base+2, c11=c_base+3
        
        # M1 = (a00+a11)(b00+b11)
        # U: a00=+1, a11=+1; V: b00=+1, b11=+1; W: c00=+1, c11=+1
        U_mat[a_base][col_start] = 1
        U_mat[a_base+3][col_start] = 1
        V_mat[b_base][col_start] = 1
        V_mat[b_base+3][col_start] = 1
        W_mat[c_base][col_start] = 1
        W_mat[c_base+3][col_start] = 1
        col_start += 1
        
        # M2 = (a10+a11)b00
        # U: a10=+1, a11=+1; V: b00=+1; W: c10=+1, c11=-1
        U_mat[a_base+2][col_start] = 1
        U_mat[a_base+3][col_start] = 1
        V_mat[b_base][col_start] = 1
        W_mat[c_base+2][col_start] = 1
        W_mat[c_base+3][col_start] = -1
        col_start += 1
        
        # M3 = a00(b10-b11)
        # U: a00=+1; V: b10=+1, b11=-1; W: c01=+1, c11=+1
        U_mat[a_base][col_start] = 1
        V_mat[b_base+2][col_start] = 1
        V_mat[b_base+3][col_start] = -1
        W_mat[c_base+1][col_start] = 1
        W_mat[c_base+3][col_start] = 1
        col_start += 1
        
        # M4 = a11(b01-b00)
        # U: a11=+1; V: b01=+1, b00=-1; W: c00=+1, c10=+1
        U_mat[a_base+3][col_start] = 1
        V_mat[b_base+1][col_start] = 1
        V_mat[b_base][col_start] = -1
        W_mat[c_base][col_start] = 1
        W_mat[c_base+2][col_start] = 1
        col_start += 1
        
        # M5 = (a00+a01)b11
        # U: a00=+1, a01=+1; V: b11=+1; W: c00=-1, c01=+1
        U_mat[a_base][col_start] = 1
        U_mat[a_base+1][col_start] = 1
        V_mat[b_base+3][col_start] = 1
        W_mat[c_base][col_start] = -1
        W_mat[c_base+1][col_start] = 1
        col_start += 1
        
        # M6 = (a10-a00)(b00+b10)
        # U: a10=+1, a00=-1; V: b00=+1, b10=+1; W: c11=+1
        U_mat[a_base+2][col_start] = 1
        U_mat[a_base][col_start] = -1
        V_mat[b_base][col_start] = 1
        V_mat[b_base+2][col_start] = 1
        W_mat[c_base+3][col_start] = 1
        col_start += 1
        
        # M7 = (a01-a11)(b01+b11)
        # U: a01=+1, a11=-1; V: b01=+1, b11=+1; W: c00=+1
        U_mat[a_base+1][col_start] = 1
        U_mat[a_base+3][col_start] = -1
        V_mat[b_base+1][col_start] = 1
        V_mat[b_base+3][col_start] = 1
        W_mat[c_base][col_start] = 1
        col_start += 1
        
        return col_start
    
    # Helper: Schoolbook 2x2x3 decomposition (rank 12)
    # A is 2x2, B is 2x3, C is 2x3
    def schoolbook_223(U_mat, V_mat, W_mat, col_start, a_base, b_base, c_base):
        """
        Add schoolbook 2x2x3 terms.
        a_base: offset for A (2x2)
        b_base: offset for B (2x3)
        c_base: offset for C (2x3)
        Returns col_start + 12
        """
        # A indices: a00=a_base, a01=a_base+1, a10=a_base+2, a11=a_base+3
        # B indices: b00=b_base, b01=b_base+1, b02=b_base+2, b10=b_base+3, b11=b_base+4, b12=b_base+5
        # C indices: c00=c_base, c01=c_base+1, c02=c_base+2, c10=c_base+3, c11=c_base+4, c12=c_base+5
        
        for a in range(2):
            for j in range(2):
                for b in range(3):
                    A_idx = a_base + a*2 + j
                    B_idx = b_base + j*3 + b
                    C_idx = c_base + a*3 + b
                    U_mat[A_idx][col_start] = 1
                    V_mat[B_idx][col_start] = 1
                    W_mat[C_idx][col_start] = 1
                    col_start += 1
        
        return col_start
    
    # Now build the full decomposition:
    # A is 2x4: split into A1 (cols 0-1) and A2 (cols 2-3)
    # A1 indices: a=0: 0,1; a=1: 2,3 -> A1 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # A2 indices: a=0: 4,5; a=1: 6,7 -> A2 = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
    
    # B is 4x7: split into B1 (rows 0-1) and B2 (rows 2-3)
    # B1 is 2x7: indices 0-6
    # B2 is 2x7: indices 7-13
    
    # C is 2x7: indices 0-13
    # C = A1*B1 + A2*B2
    
    # A1*B1: 2x2 * 2x7
    # Split B1 into B1a (2x2, cols 0-1), B1b (2x2, cols 2-3), B1c (2x3, cols 4-6)
    # A1*B1a: 2x2*2x2, Strassen rank 7
    # A1*B1b: 2x2*2x2, Strassen rank 7
    # A1*B1c: 2x2*2x3, schoolbook rank 12
    
    # A2*B2: similarly
    # B2a (2x2, cols 0-1), B2b (2x2, cols 2-3), B2c (2x3, cols 4-6)
    # A2*B2a: Strassen 7
    # A2*B2b: Strassen 7
    # A2*B2c: schoolbook 12
    
    # Total: 7+7+12+7+7+12 = 52
    
    # A1*B1a:
    # A1 base: 0 (A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3)
    # B1a base: 0 (B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3)
    # C base: 0 (C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3)
    col = strassen_222(U, V, W, col, a_base=0, b_base=0, c_base=0)
    
    # A1*B1b:
    # A1 base: 0
    # B1b base: 2 (B[0,2]=2, B[0,3]=3, B[1,2]=4, B[1,3]=5)
    # C base: 2 (C[0,2]=2, C[0,3]=3, C[1,2]=4, C[1,3]=5)
    col = strassen_222(U, V, W, col, a_base=0, b_base=2, c_base=2)
    
    # A1*B1c:
    # A1 base: 0
    # B1c base: 4 (B[0,4]=4, B[0,5]=5, B[0,6]=6, B[1,4]=7, B[1,5]=8, B[1,6]=9)
    # C base: 4 (C[0,4]=4, C[0,5]=5, C[0,6]=6, C[1,4]=7, C[1,5]=8, C[1,6]=9)
    col = schoolbook_223(U, V, W, col, a_base=0, b_base=4, c_base=4)
    
    # A2*B2a:
    # A2 base: 4 (A[0,2]=4, A[0,3]=5, A[1,2]=6, A[1,3]=7)
    # B2a base: 7 (B[2,0]=7, B[2,1]=8, B[3,0]=9, B[3,1]=10)
    # C base: 0 (C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3)
    col = strassen_222(U, V, W, col, a_base=4, b_base=7, c_base=0)
    
    # A2*B2b:
    # A2 base: 4
    # B2b base: 9 (B[2,2]=9, B[2,3]=10, B[3,2]=11, B[3,3]=12)
    # C base: 2 (C[0,2]=2, C[0,3]=3, C[1,2]=4, C[1,3]=5)
    col = strassen_222(U, V, W, col, a_base=4, b_base=9, c_base=2)
    
    # A2*B2c:
    # A2 base: 4
    # B2c base: 11 (B[2,4]=11, B[2,5]=12, B[2,6]=13, B[3,4]=14, B[3,5]=15, B[3,6]=16)
    # C base: 4 (C[0,4]=4, C[0,5]=5, C[0,6]=6, C[1,4]=7, C[1,5]=8, C[1,6]=9)
    col = schoolbook_223(U, V, W, col, a_base=4, b_base=11, c_base=4)
    
    return {"U": U, "V": V, "W": W}
