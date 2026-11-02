# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the tensor decomposition for matrix multiplication using Kronecker product structure.
    Decomposes 2x5x5 into (2x1x1) and (1x5x5).
    """
    n, m, p = DIMENSIONS
    
    # Sub-problem 1: 2x1x1
    n1, m1, p1 = 2, 1, 1
    # Sub-problem 2: 1x5x5
    n2, m2, p2 = 1, 5, 5
    
    # Generate schoolbook decomposition for sub-problem 1
    rank1 = n1 * m1 * p1
    U1 = [[0] * rank1 for _ in range(n1 * m1)]
    V1 = [[0] * rank1 for _ in range(m1 * p1)]
    W1 = [[0] * rank1 for _ in range(p1 * n1)]
    
    col = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U1[i * m1 + j][col] = 1
                V1[j * p1 + k][col] = 1
                W1[k * n1 + i][col] = 1
                col += 1
                
    # Generate schoolbook decomposition for sub-problem 2
    rank2 = n2 * m2 * p2
    U2 = [[0] * rank2 for _ in range(n2 * m2)]
    V2 = [[0] * rank2 for _ in range(m2 * p2)]
    W2 = [[0] * rank2 for _ in range(p2 * n2)]
    
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U2[i * m2 + j][col] = 1
                V2[j * p2 + k][col] = 1
                W2[k * n2 + i][col] = 1
                col += 1

    # Combine using Kronecker product
    # Rank is rank1 * rank2
    rank = rank1 * rank2
    
    # Dimensions
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    # Kronecker product of decompositions
    # Index mapping:
    # U index: (i1*m1 + j1) * m2 + (i2*m2 + j2) ? 
    # Standard Kronecker index for A (n1 x m1) and B (n2 x m2) is A[i1, j1] * B[i2, j2] at [i1*n2+i2, j1*m2+j2]
    # Here we are combining the *tensors*.
    # The tensor T = T1 x T2.
    # T1 acts on indices (i1, j1, k1)
    # T2 acts on indices (i2, j2, k2)
    # Combined indices:
    # i = i1 * n2 + i2
    # j = j1 * m2 + j2
    # k = k1 * p2 + k2
    
    for r1 in range(rank1):
        for r2 in range(rank2):
            r = r1 * rank2 + r2
            
            # Find non-zero entries in U1, V1, W1 for term r1
            # Since it's schoolbook, each term has exactly one 1 in U, V, W.
            # We can just iterate through the terms.
            
            # Get the indices from the sub-decompositions
            # U1 term r1 is non-zero at (i1, j1)
            # V1 term r1 is non-zero at (j1, k1)
            # W1 term r1 is non-zero at (k1, i1)
            
            # We need to extract i1, j1, k1 from r1.
            # In schoolbook, r1 = i1 * (m1*p1) + j1 * p1 + k1
            i1 = r1 // (m1 * p1)
            rem1 = r1 % (m1 * p1)
            j1 = rem1 // p1
            k1 = rem1 % p1
            
            # Get the indices from the sub-decompositions for r2
            # In schoolbook, r2 = i2 * (m2*p2) + j2 * p2 + k2
            i2 = r2 // (m2 * p2)
            rem2 = r2 % (m2 * p2)
            j2 = rem2 // p2
            k2 = rem2 % p2
            
            # Combined indices
            i = i1 * n2 + i2
            j = j1 * m2 + j2
            k = k1 * p2 + k2
            
            # U index: i * m + j
            U[i * m + j][r] = 1
            # V index: j * p + k
            V[j * p + k][r] = 1
            # W index: k * n + i
            W[k * n + i][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
