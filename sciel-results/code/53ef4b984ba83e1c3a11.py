"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition for 4x4 matrix multiplication
    # Rank 49 (7x7)
    
    # Define 2x2 Strassen factors
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices for n=2:
    # U row: i*2 + j
    # V row: j*2 + k
    # W row: k*2 + i
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # i=0, j=0 (A00)
        [0, 0, 0, 0, 1, 0, 1],  # i=0, j=1 (A01)
        [0, 1, 0, 0, 0, 1, 0],  # i=1, j=0 (A10)
        [1, 1, 0, 1, 0, 0, -1]  # i=1, j=1 (A11)
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # j=0, k=0 (B00)
        [0, 0, 1, 0, 0, 1, 0],  # j=0, k=1 (B01)
        [0, 0, 0, 1, 0, 0, 1],  # j=1, k=0 (B10)
        [1, 0, -1, 0, 1, 0, 1]  # j=1, k=1 (B11)
    ]
    
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],  # k=0, i=0 (C00)
        [0, 1, 0, 1, 0, 0, 0],   # k=0, i=1 (C10)
        [0, 0, 1, 0, 1, 0, 0],   # k=1, i=0 (C01)
        [1, -1, 1, 0, 0, 1, 0]   # k=1, i=1 (C11)
    ]
    
    n, m, p = DIMENSIONS
    # n=4, m=4, p=4
    # Rank 49
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Tensor product construction
    # U4[i*4+j, r1*7+r2] = U2[i1*2+j1, r1] * U2[i0*2+j0, r2]
    # where i = 2*i1 + i0, j = 2*j1 + j0
    
    for i in range(4):
        for j in range(4):
            i1, i0 = divmod(i, 2)
            j1, j0 = divmod(j, 2)
            u_row_1 = i1 * 2 + j1
            u_row_0 = i0 * 2 + j0
            
            for r1 in range(7):
                for r2 in range(7):
                    col = r1 * 7 + r2
                    val = U2[u_row_1][r1] * U2[u_row_0][r2]
                    if val != 0:
                        U[i * 4 + j][col] = val
                        
    for j in range(4):
        for k in range(4):
            j1, j0 = divmod(j, 2)
            k1, k0 = divmod(k, 2)
            v_row_1 = j1 * 2 + k1
            v_row_0 = j0 * 2 + k0
            
            for r1 in range(7):
                for r2 in range(7):
                    col = r1 * 7 + r2
                    val = V2[v_row_1][r1] * V2[v_row_0][r2]
                    if val != 0:
                        V[j * 4 + k][col] = val
                        
    for k in range(4):
        for i in range(4):
            k1, k0 = divmod(k, 2)
            i1, i0 = divmod(i, 2)
            w_row_1 = k1 * 2 + i1
            w_row_0 = k0 * 2 + i0
            
            for r1 in range(7):
                for r2 in range(7):
                    col = r1 * 7 + r2
                    val = W2[w_row_1][r1] * W2[w_row_0][r2]
                    if val != 0:
                        W[k * 4 + i][col] = val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
