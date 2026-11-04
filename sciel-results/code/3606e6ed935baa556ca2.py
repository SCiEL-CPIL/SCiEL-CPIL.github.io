"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition for 2x2 matrix multiplication (rank 7)
    # U: shape (4, 7), indexed by [i*2+j, col]
    # V: shape (4, 7), indexed by [j*2+k, col]
    # W: shape (4, 7), indexed by [k*2+i, col] (trace convention)
    
    U1 = [
        [1, 0, 1, 0, 1, -1, 0],   # A11
        [0, 0, 0, 0, 1, 0, -1],   # A12
        [0, 1, 0, 0, 0, 1, 0],    # A21
        [1, 1, 0, 1, 0, 0, 1]     # A22
    ]
    
    V1 = [
        [1, 1, 0, -1, 0, 1, 0],   # B11
        [0, 0, 0, 0, 0, 1, 0],    # B12
        [1, 0, 1, 0, 0, 0, 1],    # B21
        [1, 0, -1, 0, 1, 0, 1]    # B22
    ]
    
    W1 = [
        [1, 0, 0, -1, 1, 0, -1],  # C11 (k=0,i=0)
        [0, 0, 1, 0, 1, 0, 0],    # C12 (k=0,i=1)
        [0, 1, 0, 1, 0, 0, 0],    # C21 (k=1,i=0)
        [1, -1, 1, 0, 0, 1, 0]    # C22 (k=1,i=1)
    ]
    
    # Tensor product of two Strassen decompositions for 4x4
    n1, m1, p1 = 2, 2, 2
    n2, m2, p2 = 2, 2, 2
    r1 = 7
    r2 = 7
    rank = r1 * r2  # 49
    
    # U_combined: shape (n1*m1*n2*m2, rank) = (16, 49)
    # Row index: (i1*m1+j1)*n2*m2 + (i2*m2+j2)
    # Col index: r1_idx * r2 + r2_idx
    U = [[0] * rank for _ in range(n1 * m1 * n2 * m2)]
    for i1 in range(n1):
        for j1 in range(m1):
            for i2 in range(n2):
                for j2 in range(m2):
                    row = (i1 * m1 + j1) * n2 * m2 + (i2 * m2 + j2)
                    for r1_idx in range(r1):
                        for r2_idx in range(r2):
                            col = r1_idx * r2 + r2_idx
                            U[row][col] = U1[i1 * m1 + j1][r1_idx] * U2_val if False else U1[i1 * m1 + j1][r1_idx] * U1[i2 * m2 + j2][r2_idx]  # Using same decomposition for both
    # Wait, I'm using the same Strassen decomposition for both tensor factors, so U2 = U1, V2 = V1, W2 = W1
    
    # Let me redo this more cleanly
    U = [[0] * rank for _ in range(n1 * m1 * n2 * m2)]
    V = [[0] * rank for _ in range(m1 * p1 * m2 * p2)]
    W = [[0] * rank for _ in range(p1 * n1 * p2 * n2)]
    
    for i1 in range(n1):
        for j1 in range(m1):
            for i2 in range(n2):
                for j2 in range(m2):
                    row_u = (i1 * m1 + j1) * n2 * m2 + (i2 * m2 + j2)
                    for r1_idx in range(r1):
                        for r2_idx in range(r2):
                            col = r1_idx * r2 + r2_idx
                            U[row_u][col] = U1[i1 * m1 + j1][r1_idx] * U1[i2 * m2 + j2][r2_idx]
    
    for j1 in range(m1):
        for k1 in range(p1):
            for j2 in range(m2):
                for k2 in range(p2):
                    row_v = (j1 * p1 + k1) * m2 * p2 + (j2 * p2 + k2)
                    for r1_idx in range(r1):
                        for r2_idx in range(r2):
                            col = r1_idx * r2 + r2_idx
                            V[row_v][col] = V1[j1 * p1 + k1][r1_idx] * V1[j2 * p2 + k2][r2_idx]
    
    for k1 in range(p1):
        for i1 in range(n1):
            for k2 in range(p2):
                for i2 in range(n2):
                    row_w = (k1 * n1 + i1) * p2 * n2 + (k2 * n2 + i2)
                    for r1_idx in range(r1):
                        for r2_idx in range(r2):
                            col = r1_idx * r2 + r2_idx
                            W[row_w][col] = W1[k1 * n1 + i1][r1_idx] * W1[k2 * n2 + i2][r2_idx]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
