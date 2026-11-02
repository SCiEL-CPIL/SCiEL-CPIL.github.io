"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # U, V, W are 4x7 matrices for one block.
    # We will add these to the global U, V, W for each block.
    
    # Standard Strassen decomposition
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B22)
    # P4 = A22(B12-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # C11 = P1 - P4 + P5 - P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 - P4
    
    # Indices:
    # U: 0:A11, 1:A12, 2:A21, 3:A22
    # V: 0:B11, 1:B12, 2:B21, 3:B22
    # W: 0:C11, 1:C21, 2:C12, 3:C22 (Note: W index k*n+i. For 2x2 block, k is 0 or 1 relative to block start? No, k is global row index of C.
    # If block is at cols j, j+1.
    # C rows are 0, 1.
    # C[0,j] -> W[j*n+0]
    # C[1,j] -> W[j*n+1]
    # C[0,j+1] -> W[(j+1)*n+0]
    # C[1,j+1] -> W[(j+1)*n+1]
    
    # So for block b (0-indexed), cols are 2*b and 2*b+1.
    # W indices:
    # C11 (0, 2b) -> W[2b*n + 0]
    # C21 (1, 2b) -> W[2b*n + 1]
    # C12 (0, 2b+1) -> W[(2b+1)*n + 0]
    # C22 (1, 2b+1) -> W[(2b+1)*n + 1]
    
    strassen_U = [
        [1, 0, 1, 0, 1, -1, 0], # P1, P2, P3, P4, P5, P6, P7
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    
    strassen_V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    
    strassen_W = [
        [1, 0, 0, -1, 1, 0, -1], # C11
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [1, -1, 1, -1, 0, 0, 0]  # C22
    ]
    
    for b in range(7):
        base_col = b * 7
        v_offset = b * 2
        w_offset = b * 2 * n
        
        for t in range(7):
            col = base_col + t
            
            # U
            for i in range(4):
                U[i][col] += strassen_U[i][t]
                
            # V
            for j in range(4):
                V[v_offset + j][col] += strassen_V[j][t]
                
            # W
            for k in range(4):
                # W index mapping:
                # k=0 -> C11 -> W[w_offset + 0]
                # k=1 -> C21 -> W[w_offset + 1]
                # k=2 -> C12 -> W[w_offset + n]
                # k=3 -> C22 -> W[w_offset + n + 1]
                
                if k == 0:
                    W[w_offset][col] += strassen_W[0][t]
                elif k == 1:
                    W[w_offset + 1][col] += strassen_W[1][t]
                elif k == 2:
                    W[w_offset + n][col] += strassen_W[2][t]
                elif k == 3:
                    W[w_offset + n + 1][col] += strassen_W[3][t]

    # Last block: 2x1
    # Cols 14
    # V indices: 14, 15
    # W indices: 14*n+0, 14*n+1
    
    col = 49
    # 1. A00 B0
    U[0][col] = 1
    V[14][col] = 1
    W[14*n][col] = 1
    col += 1
    
    # 2. A01 B1
    U[1][col] = 1
    V[15][col] = 1
    W[14*n][col] = 1
    col += 1
    
    # 3. A10 B0
    U[2][col] = 1
    V[14][col] = 1
    W[14*n+1][col] = 1
    col += 1
    
    # 4. A11 B1
    U[3][col] = 1
    V[15][col] = 1
    W[14*n+1][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
