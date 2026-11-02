import time

DIMENSIONS = [4, 4, 5]

def construct():
    """
    Returns a tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Uses the recursive Strassen decomposition for the 4x4x4 part (rank 49)
    and schoolbook for the 4x4x1 part (rank 16).
    
    Total rank: 49 + 16 = 65.
    
    The matrices U, V, W have shapes:
      U: (n*m, rank) = (16, 65)
      V: (m*p, rank) = (20, 65)
      W: (p*n, rank) = (20, 65)
    
    Entry convention:
      U[i*m+j, c] for i in range(n), j in range(m)
      V[j*p+k, c] for j in range(m), k in range(p)
      W[k*n+i, c] for k in range(p), i in range(n)
    """
    n, m, p = 4, 4, 5
    
    # Strassen 2x2x2 decomposition
    # U, V, W are lists of 7 matrices (each 2x2)
    # Term r: C block r = A_block1(r) * A_block2(r)
    # where A_block1, A_block2 are linear combos of A blocks, B blocks
    
    # Strassen terms:
    # 0: (A00+B11)(B00+B11)  -> M0
    # 1: (A10-A00)B00        -> M1
    # 2: A00(B01-B11)        -> M2
    # 3: (A01+A10)(B00+B11)  -> M3
    # 4: (A00-A10)(B10+B11)  -> M4
    # 5: (A01-A11)(B00+B10)  -> M5
    # 6: (A11+A00)(B11-B01)  -> M6
    
    # C00 = M0 + M3 - M4 - M6
    # C01 = M2 + M3
    # C10 = M1 + M4
    # C11 = M0 - M1 + M5 + M6
    
    # For each Strassen term, we have factor matrices U_r (2x2), V_r (2x2), W_r (2x2)
    # such that the contribution to C block (i,j) is sum over r: W_r[i,j] * (U_r * A_block) * (V_r * B_block)
    
    # Let's define the Strassen factors explicitly.
    # For term r, A_block = sum over (a,b) of U_r[a,b] * A[a,b]
    #                  B_block = sum over (c,d) of V_r[c,d] * B[c,d]
    # And C_block[i,j] += W_r[i,j] * (A_block * B_block)
    
    # Strassen term definitions:
    # r=0: A_block = A00 + A11, B_block = B00 + B11, W = diag(1, 0, 0, 1) -> W[0,0]=1, W[1,1]=1
    # r=1: A_block = A10 - A00, B_block = B00, W = diag(0, 0, 1, 0) -> W[1,0]=1
    # r=2: A_block = A00, B_block = B01 - B11, W = diag(0, 1, 0, 0) -> W[0,1]=1
    # r=3: A_block = A01 + A10, B_block = B00 + B11, W = diag(1, 1, 0, 0) -> W[0,0]=1, W[0,1]=1
    # r=4: A_block = A00 - A10, B_block = B10 + B11, W = diag(0, 0, 1, 0) -> W[1,0]=1  (but with sign)
    # Actually, C10 = M1 + M4 where M4 = (A00-A10)(B10+B11)
    # r=5: A_block = A01 - A11, B_block = B00 + B10, W = diag(0, 0, 0, 1) -> W[1,1]=1
    # r=6: A_block = A11 + A00, B_block = B11 - B01, W = diag(1, 0, 0, 1) -> W[0,0]=-1, W[1,1]=1
    
    # Let me be more careful. The standard Strassen decomposition:
    # M0 = (A00 + A11)(B00 + B11)
    # M1 = (A10 - A00)B00
    # M2 = A00(B01 - B11)
    # M3 = (A01 + A10)(B00 + B11)
    # M4 = (A00 - A10)(B10 + B11)
    # M5 = (A01 - A11)(B00 + B10)
    # M6 = (A11 + A00)(B11 - B01)
    
    # C00 = M0 + M3 - M4 - M6
    # C01 = M2 + M3
    # C10 = M1 + M4
    # C11 = M0 - M1 + M5 + M6
    
    # So for each output block (i,j), the coefficient of M_r is:
    # W_matrix[r][i][j]
    
    strassen_U = [
        [[1,0],[0,1]],  # M0: A00+A11
        [[0,-1],[1,0]], # M1: A10-A00
        [[1,0],[0,0]],  # M2: A00
        [[0,1],[1,0]],  # M3: A01+A10
        [[1,0],[-1,0]], # M4: A00-A10
        [[0,1],[0,-1]], # M5: A01-A11
        [[1,0],[0,1]],  # M6: A11+A00
    ]
    strassen_V = [
        [[1,0],[0,1]],  # M0: B00+B11
        [[1,0],[0,0]],  # M1: B00
        [[0,1],[0,-1]], # M2: B01-B11
        [[1,0],[0,1]],  # M3: B00+B11
        [[0,1],[1,1]],  # M4: B10+B11
        [[1,1],[0,0]],  # M5: B00+B10
        [[0,-1],[0,1]], # M6: B11-B01
    ]
    strassen_W = [
        [[1,0],[0,1]],  # M0: C00 += M0, C11 += M0
        [[0,0],[1,0]],  # M1: C10 += M1
        [[0,1],[0,0]],  # M2: C01 += M2
        [[1,1],[0,0]],  # M3: C00 += M3, C01 += M3
        [[0,0],[1,0]],  # M4: C10 += M4
        [[0,0],[0,1]],  # M5: C11 += M5
        [[-1,0],[0,1]], # M6: C00 -= M6, C11 += M6
    ]
    
    rank = 49 + 16
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Fill in the Strassen 4x4x4 part (first 49 columns)
    col = 0
    for r_outer in range(7):
        for r_inner in range(7):
            # U matrix for this term: strassen_U[r_outer] (2x2) kron strassen_U[r_inner] (2x2) -> 4x4
            # V matrix: strassen_V[r_outer] (2x2) kron strassen_V[r_inner] (2x2) -> 4x4
            # W matrix: strassen_W[r_outer] (2x2) kron strassen_W[r_inner] (2x2) -> 4x4
            
            # Fill U: U[i*m+j, col] where i,j in range(4)
            # The 4x4 matrix is kron of two 2x2 matrices
            for i in range(4):
                for j in range(4):
                    i_out, i_in = i // 2, i % 2
                    j_out, j_in = j // 2, j % 2
                    val = strassen_U[r_outer][i_out][j_out] * strassen_U[r_inner][i_in][j_in]
                    U[i * m + j][col] = val
            
            # Fill V: V[j*p+k, col] where j in range(m=4), k in range(p=5)
            # For the 4x4x4 part, k is in range(4). The 5th column (k=4) is handled separately.
            for j in range(4):
                for k in range(4):
                    j_out, j_in = j // 2, j % 2
                    k_out, k_in = k // 2, k % 2
                    val = strassen_V[r_outer][j_out][k_out] * strassen_V[r_inner][j_in][k_in]
                    V[j * p + k][col] = val
            
            # Fill W: W[k*n+i, col] where k in range(p=5), i in range(n=4)
            # For the 4x4x4 part, k is in range(4).
            for k in range(4):
                for i in range(4):
                    k_out, k_in = k // 2, k % 2
                    i_out, i_in = i // 2, i % 2
                    val = strassen_W[r_outer][k_out][i_out] * strassen_W[r_inner][k_in][i_in]
                    W[k * n + i][col] = val
            
            col += 1
    
    # Now fill in the schoolbook 4x4x1 part (last 16 columns)
    # This handles the 5th column of B (k=4)
    for i in range(4):
        for j in range(4):
            # U[i*m+j, col] = 1
            U[i * m + j][col] = 1.0
            # V[j*p+4, col] = 1
            V[j * p + 4][col] = 1.0
            # W[4*n+i, col] = 1
            W[4 * n + i][col] = 1.0
            col += 1
    
    return {"U": U, "V": V, "W": W}
