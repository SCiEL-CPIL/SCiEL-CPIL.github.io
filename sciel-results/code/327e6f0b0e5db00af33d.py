def _block_2_1_2(j, k0, k1, U, V, W, col, n, m, p):
    # Term 1: A[j] * (B[k0] + B[k1]) -> C[0, k0] += A[j]B[k0], C[0, k1] += A[j]B[k1]
    # U[j] = 1
    # V[k0] = 1, V[k1] = 1
    # W[k0, 0] = 1, W[k1, 0] = 1
    U[0*m + j][col] += 1
    V[k0*p + j][col] += 1  # Wait, V index is l*p + k. Here l=j? No.
    # A is n x m. B is m x p.
    # j is the index in m.
    # So V index is j*p + k.
    V[j*p + k0][col] += 1
    V[j*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    W[k1*n + 0][col] += 1
    col += 1
    
    # Term 2: A[j] * (B[k0] + B[k1]) -> C[1, k0] += A[j]B[k0], C[1, k1] += A[j]B[k1]
    # Wait, A has 2 rows.
    # Term 1 was for Row 0 of A?
    # No, A[j] is a vector of length n=2.
    # My previous derivation:
    # P1 = A_0 (B_0+B_1). A_0 is the first row of A.
    # So U[0, j] = 1.
    # P2 = A_1 (B_0+B_1). A_1 is the second row of A.
    # So U[1, j] = 1.
    
    U[1*m + j][col] += 1
    V[j*p + k0][col] += 1
    V[j*p + k1][col] += 1
    W[k0*n + 1][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    return col
