"""Kronecker-product-based rank-42 decomposition for 2x4 by 4x6 matrix multiplication."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # We decompose the 2x4 x 4x6 multiplication tensor as a Kronecker product of
    # a 2x2 x 2 tensor (Strassen, rank 7) and a 1x2 x 3 tensor (trivial, rank 6).
    #
    # The overall dimensions are:
    #   n = 2, m = 4, p = 6
    # We split:
    #   n = n1 * n2 = 2 * 1
    #   m = m1 * m2 = 2 * 2
    #   p = p1 * p2 = 2 * 3
    #
    # So the first tensor is (n1, m1, p1) = (2, 2, 2) with rank 7 (Strassen).
    # The second tensor is (n2, m2, p2) = (1, 2, 3) with rank 6 (schoolbook).
    # Total rank = 7 * 6 = 42.
    #
    # Index conventions (following the trace convention W[k*n + i, r]):
    #   U has shape (n*m, rank) = (8, 42), indexed by (i*m + j, c)
    #   V has shape (m*p, rank) = (24, 42), indexed by (j*p + k, c)
    #   W has shape (p*n, rank) = (12, 42), indexed by (k*n + i, c)
    #
    # Where i in [0,n), j in [0,m), k in [0,p), c in [0,rank).
    #
    # We decompose indices as:
    #   i = i1 + i2 * n1  ->  i1 in [0,n1), i2 in [0,n2)
    #   j = j1 + j2 * m1  ->  j1 in [0,m1), j2 in [0,m2)
    #   k = k1 + k2 * p1  ->  k1 in [0,p1), k2 in [0,p2)
    #   c = c1 + c2 * rank1 -> c1 in [0,rank1), c2 in [0,rank2)
    #
    # The decomposition is:
    #   U[i*m + j, c] = U1[i1*m1 + j1, c1] * U2[i2*m2 + j2, c2]
    #   V[j*p + k, c] = V1[j1*p1 + k1, c1] * V2[j2*p2 + k2, c2]
    #   W[k*n + i, c] = W1[k1*n1 + i1, c1] * W2[k2*n2 + i2, c2]
    
    n, m, p = DIMENSIONS
    n1, m1, p1 = 2, 2, 2
    n2, m2, p2 = 1, 2, 3
    rank1, rank2 = 7, 6
    total_rank = rank1 * rank2
    
    # Initialize matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen-like decomposition for 2x2 x 2x2 (rank 7)
    # We need U1, V1, W1 for the (2,2,2) tensor.
    # The standard Strassen decomposition gives:
    #   M1 = (A11 - A22)(B22 - B11)
    #   M2 = (A21 + A11)B11
    #   M3 = A12(B22 - B12)
    #   M4 = (A11 - A12 - A22)(B11 + B22)
    #   M5 = (A11 + A21)(B22 - B21)
    #   M6 = (A12 + A22 - A11 - A21)(B11 + B12)
    #   M7 = (A11 - A12 + A21 - A22)(B12 + B22)
    #
    # For the tensor decomposition, we express:
    #   C11 = M4 + M5 - M7 + M1
    #   C12 = M3 + M7
    #   C21 = M2 + M6
    #   C22 = M4 - M1 + M2 - M6
    #
    # Let's build U1, V1, W1 directly from the bilinear forms.
    # U1 has shape (n1*m1, rank1) = (4, 7)
    # V1 has shape (m1*p1, rank1) = (4, 7)
    # W1 has shape (p1*n1, rank1) = (4, 7)
    
    U1 = [[0] * rank1 for _ in range(n1 * m1)]
    V1 = [[0] * rank1 for _ in range(m1 * p1)]
    W1 = [[0] * rank1 for _ in range(p1 * n1)]
    
    # Strassen's 7 products:
    # Product 0: M1 = (A11 - A22)(B22 - B11)
    U1[0][0] = 1   # A11
    U1[3][0] = -1  # A22
    V1[3][0] = 1   # B22
    V1[0][0] = -1  # B11
    W1[3][0] = 1   # C11
    W1[3][0] = 1   # C22 (same index since n1=p1=2, k*n1+i: k=1,i=1 -> 3; k=1,i=1 -> 3)
    # Wait, let me re-check the W indexing. W[k*n1 + i].
    # C11: k=0, i=0 -> index 0
    # C12: k=0, i=1 -> index 1
    # C21: k=1, i=0 -> index 2
    # C22: k=1, i=1 -> index 3
    
    # Let me redo this more carefully.
    
    # Reset
    U1 = [[0] * rank1 for _ in range(n1 * m1)]
    V1 = [[0] * rank1 for _ in range(m1 * p1)]
    W1 = [[0] * rank1 for _ in range(p1 * n1)]
    
    # Indices:
    # U1[i1*m1 + j1, c1]: i1 in [0,2), j1 in [0,2)
    #   (0,0)=A11 -> 0, (0,1)=A12 -> 1, (1,0)=A21 -> 2, (1,1)=A22 -> 3
    # V1[j1*p1 + k1, c1]: j1 in [0,2), k1 in [0,2)
    #   (0,0)=B11 -> 0, (0,1)=B12 -> 1, (1,0)=B21 -> 2, (1,1)=B22 -> 3
    # W1[k1*n1 + i1, c1]: k1 in [0,2), i1 in [0,2)
    #   (0,0)=C11 -> 0, (0,1)=C12 -> 1, (1,0)=C21 -> 2, (1,1)=C22 -> 3
    
    # M1 = (A11 - A22)(B22 - B11) -> contributes to C11 and C22
    # C11 += M1, C22 += M1
    U1[0][0] += 1;  U1[3][0] += -1
    V1[3][0] += 1;  V1[0][0] += -1
    W1[0][0] += 1;  W1[3][0] += 1
    
    # M2 = (A21 + A11)B11 -> contributes to C21 and C22
    # C21 += M2, C22 += M2
    U1[2][1] += 1;  U1[0][1] += 1
    V1[0][1] += 1
    W1[2][1] += 1;  W1[3][1] += 1
    
    # M3 = A12(B22 - B12) -> contributes to C12
    # C12 += M3
    U1[1][2] += 1
    V1[3][2] += 1;  V1[1][2] += -1
    W1[1][2] += 1
    
    # M4 = (A11 - A12 - A22)(B11 + B22) -> contributes to C11 and C22
    # C11 += M4, C22 += M4
    U1[0][3] += 1;  U1[1][3] += -1;  U1[3][3] += -1
    V1[0][3] += 1;  V1[3][3] += 1
    W1[0][3] += 1;  W1[3][3] += 1
    
    # M5 = (A11 + A21)(B22 - B21) -> contributes to C11 and C21
    # C11 += M5, C21 += M5
    U1[0][4] += 1;  U1[2][4] += 1
    V1[3][4] += 1;  V1[2][4] += -1
    W1[0][4] += 1;  W1[2][4] += 1
    
    # M6 = (A12 + A22 - A11 - A21)(B11 + B12) -> contributes to C21
    # C21 += M6
    U1[1][5] += 1;  U1[3][5] += 1;  U1[0][5] += -1;  U1[2][5] += -1
    V1[0][5] += 1;  V1[1][5] += 1
    W1[2][5] += 1
    
    # M7 = (A11 - A12 + A21 - A22)(B12 + B22) -> contributes to C11 and C12
    # C11 += M7, C12 += M7
    U1[0][6] += 1;  U1[1][6] += -1;  U1[2][6] += 1;  U1[3][6] += -1
    V1[1][6] += 1;  V1[3][6] += 1
    W1[0][6] += 1;  W1[1][6] += 1
    
    # Now the trivial (1,2,3) tensor decomposition with rank 6.
    # n2=1, m2=2, p2=3
    # U2 has shape (n2*m2, rank2) = (2, 6)
    # V2 has shape (m2*p2, rank2) = (6, 6)
    # W2 has shape (p2*n2, rank2) = (3, 6)
    
    U2 = [[0] * rank2 for _ in range(n2 * m2)]
    V2 = [[0] * rank2 for _ in range(m2 * p2)]
    W2 = [[0] * rank2 for _ in range(p2 * n2)]
    
    # Schoolbook: for each (j2, k2), one product
    # j2 in [0,2), k2 in [0,3)
    # U2[i2*m2 + j2, d] = 1 if i2=0 (since n2=1)
    # V2[j2*p2 + k2, d] = 1
    # W2[k2*n2 + i2, d] = 1 if i2=0
    
    d = 0
    for j2 in range(m2):
        for k2 in range(p2):
            U2[j2][d] = 1
            V2[j2 * p2 + k2][d] = 1
            W2[k2][d] = 1
            d += 1
    
    # Now assemble the full U, V, W
    for c1 in range(rank1):
        for c2 in range(rank2):
            c = c1 + c2 * rank1
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            i = i1 + i2 * n1
                            j = j1 + j2 * m1
                            val_u = U1[i1 * m1 + j1][c1] * U2[i2 * m2 + j2][c2]
                            if val_u != 0:
                                U[i * m + j][c] += val_u
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            j = j1 + j2 * m1
                            k = k1 + k2 * p1
                            val_v = V1[j1 * p1 + k1][c1] * V2[j2 * p2 + k2][c2]
                            if val_v != 0:
                                V[j * p + k][c] += val_v
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            k = k1 + k2 * p1
                            i = i1 + i2 * n1
                            val_w = W1[k1 * n1 + i1][c1] * W2[k2 * n2 + i2][c2]
                            if val_w != 0:
                                W[k * n + i][c] += val_w
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
