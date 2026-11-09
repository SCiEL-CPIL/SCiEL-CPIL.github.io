import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4x4 Strassen decomposition (Rank 49)
    # Part 2: 4x4x3 Schoolbook decomposition (Rank 48)
    
    rank1 = 49
    rank2 = 48
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Strassen 2x2x2 Factors ---
    # Indices: i,j in 0..1 for A, j,k in 0..1 for B, k,i in 0..1 for C
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # 7 terms
    strassen_U = np.zeros((4, 7))
    strassen_V = np.zeros((4, 7))
    strassen_W = np.zeros((4, 7))
    
    # Term 0: M1 = (A11+A22)(B11+B22)
    strassen_U[0, 0] = 1; strassen_U[3, 0] = 1
    strassen_V[0, 0] = 1; strassen_V[3, 0] = 1
    strassen_W[0, 0] = 1; strassen_W[3, 0] = 1
    
    # Term 1: M2 = (A21+A22)B11
    strassen_U[2, 1] = 1; strassen_U[3, 1] = 1
    strassen_V[0, 1] = 1
    strassen_W[2, 1] = 1; strassen_W[3, 1] = 1
    
    # Term 2: M3 = A11(B12-B22)
    strassen_U[0, 2] = 1
    strassen_V[1, 2] = 1; strassen_V[3, 2] = -1
    strassen_W[1, 2] = 1; strassen_W[3, 2] = -1
    
    # Term 3: M4 = A22(B21-B11)
    strassen_U[3, 3] = 1
    strassen_V[2, 3] = 1; strassen_V[0, 3] = -1
    strassen_W[2, 3] = 1; strassen_W[0, 3] = -1
    
    # Term 4: M5 = (A11+A12)B22
    strassen_U[0, 4] = 1; strassen_U[1, 4] = 1
    strassen_V[3, 4] = 1
    strassen_W[0, 4] = 1; strassen_W[1, 4] = 1
    
    # Term 5: M6 = (A21-A11)(B11+B12)
    strassen_U[2, 5] = 1; strassen_U[0, 5] = -1
    strassen_V[0, 5] = 1; strassen_V[1, 5] = 1
    strassen_W[2, 5] = 1; strassen_W[1, 5] = 1
    
    # Term 6: M7 = (A12-A22)(B21+B22)
    strassen_U[1, 6] = 1; strassen_U[3, 6] = -1
    strassen_V[2, 6] = 1; strassen_V[3, 6] = 1
    strassen_W[1, 6] = 1; strassen_W[3, 6] = 1
    
    # Kronecker Product for 4x4x4
    # The 4x4 matrices are 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # The indices for A in 4x4 are i*4+j.
    # Block (b_i, b_j) contains 2x2 matrix.
    # Local indices u,v in 0..1.
    # Global index for A: b_i * 2 + u for row, b_j * 2 + v for col?
    # Wait, standard row-major: index = row * cols + col.
    # For 4x4, row = b_i * 2 + u, col = b_j * 2 + v.
    # Index = (b_i * 2 + u) * 4 + (b_j * 2 + v).
    
    # Strassen U indices: i*2+j where i is block row, j is block col?
    # No, Strassen U is for 2x2. Index i*2+j.
    # In Kronecker, we have two levels.
    # Level 1 (Block): 2x2.
    # Level 2 (Local): 2x2.
    
    # U_4x4[i*4+j, c]
    # i = b_i * 2 + u, j = b_j * 2 + v
    # c = c1 * 7 + c2
    
    for b_i in range(2):
        for b_j in range(2):
            for u in range(2):
                for v in range(2):
                    idx_A = (b_i * 2 + u) * 4 + (b_j * 2 + v)
                    for c1 in range(7):
                        for c2 in range(7):
                            val = strassen_U[b_i * 2 + b_j, c1] * strassen_U[u * 2 + v, c2]
                            if val != 0:
                                U[idx_A][c1 * 7 + c2] += val
                                
    # V_4x4[j*4+k, c]
    # j = b_j * 2 + v, k = b_k * 2 + w
    for b_j in range(2):
        for b_k in range(2):
            for v in range(2):
                for w in range(2):
                    idx_B = (b_j * 2 + v) * 4 + (b_k * 2 + w)
                    for c1 in range(7):
                        for c2 in range(7):
                            val = strassen_V[b_j * 2 + b_k, c1] * strassen_V[v * 2 + w, c2]
                            if val != 0:
                                V[idx_B][c1 * 7 + c2] += val
                                
    # W_4x4[k*4+i, c]
    # k = b_k * 2 + w, i = b_i * 2 + u
    for b_k in range(2):
        for b_i in range(2):
            for w in range(2):
                for u in range(2):
                    idx_C = (b_k * 2 + w) * 4 + (b_i * 2 + u)
                    for c1 in range(7):
                        for c2 in range(7):
                            val = strassen_W[b_k * 2 + b_i, c1] * strassen_W[w * 2 + u, c2]
                            if val != 0:
                                W[idx_C][c1 * 7 + c2] += val
                                
    # --- Schoolbook 4x4x3 ---
    # A is 4x4, B is 4x3 (columns 4..6 of the original 4x7 B)
    # C is 4x3 (columns 0..2 of the second part's C, which map to 4..6 of global C)
    # Global B index: j*7 + k. Here k is 4..6. Let k' = k-4 in 0..2.
    # Global C index: k*4 + i. Here k is 4..6. Let k' = k-4.
    
    offset = rank1
    for i in range(4):
        for j in range(4):
            for k_prime in range(3):
                col = offset + i * 4 * 3 + j * 3 + k_prime
                # U[i*4+j, col] = 1
                U[i * 4 + j][col] = 1
                # V[j*7 + (4+k_prime), col] = 1
                V[j * 7 + 4 + k_prime][col] = 1
                # W[(4+k_prime)*4 + i, col] = 1
                W[(4 + k_prime) * 4 + i][col] = 1
                
    return {"U": U, "V": V, "W": W}
