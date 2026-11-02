"""Matrix multiplication 2,4,7 with rank-minimal decomposition using Strassen and block structure."""
DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (14, rank)
    
    # Split m=4 into two blocks of size 2: j in {0,1} and j in {2,3}
    # For each block, split p=7 into 3 blocks of size 2 and 1 block of size 1.
    # Use Strassen (rank 7) for 2x2 * 2x2, and schoolbook (rank 4) for 2x2 * 2x1.
    
    rank = 2 * (3 * 7 + 4)  # 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2*2x2:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)(b11+b12)
    # M3 = (a11-a22)(b21+b22)
    # M4 = a11(b21-b22)
    # M5 = a22(b12-b11)
    # M6 = (a11+a12)b22
    # M7 = (a21-a11)b11
    # C11 = M1 + M4 - M5 + M7
    # C12 = M6 + M5
    # C21 = M2 - M7
    # C22 = M1 - M4 + M3 - M6
    
    # In our indexing:
    # A[i,j] with i in {0,1}, j in {j_base, j_base+1}
    # B[j,k] with j in {j_base, j_base+1}, k in {k_base, k_base+1}
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # W index: k*n + i, so W[k_base + 0, 0] = C[0, k_base], W[k_base + 0, 1] = C[1, k_base]
    # W[k_base + 1, 0] = C[0, k_base+1], W[k_base + 1, 1] = C[1, k_base+1]
    
    # So:
    # C[0, k_base] = C11 -> W[k_base*2 + 0]
    # C[1, k_base] = C21 -> W[k_base*2 + 1]
    # C[0, k_base+1] = C12 -> W[(k_base+1)*2 + 0]
    # C[1, k_base+1] = C22 -> W[(k_base+1)*2 + 1]
    
    # M1: (a11+a22)(b11+b22)
    # a11 = A[0, j_base], a22 = A[1, j_base+1]
    # b11 = B[j_base, k_base], b22 = B[j_base+1, k_base+1]
    # U: A[0, j_base] + A[1, j_base+1]
    # V: B[j_base, k_base] + B[j_base+1, k_base+1]
    # W: C11: +1, C22: +1
    U[j_base][col] = 1
    U[4 + j_base + 1][col] = 1
    V[j_base * 7 + k_base][col] = 1
    V[(j_base + 1) * 7 + (k_base + 1)][col] = 1
    W[k_base * 2 + 0][col] = 1
    W[(k_base + 1) * 2 + 1][col] = 1
    col += 1
    
    # M2: (a21+a22)(b11+b12)
    # a21 = A[1, j_base], a22 = A[1, j_base+1]
    # b11 = B[j_base, k_base], b12 = B[j_base, k_base+1]
    # U: A[1, j_base] + A[1, j_base+1]
    # V: B[j_base, k_base] + B[j_base, k_base+1]
    # W: C21: +1
    U[4 + j_base][col] = 1
    U[4 + j_base + 1][col] = 1
    V[j_base * 7 + k_base][col] = 1
    V[j_base * 7 + (k_base + 1)][col] = 1
    W[k_base * 2 + 1][col] = 1
    col += 1
    
    # M3: (a11-a22)(b21+b22)
    # a11 = A[0, j_base], a22 = A[1, j_base+1]
    # b21 = B[j_base+1, k_base], b22 = B[j_base+1, k_base+1]
    # U: A[0, j_base] - A[1, j_base+1]
    # V: B[j_base+1, k_base] + B[j_base+1, k_base+1]
    # W: C22: +1
    U[j_base][col] = 1
    U[4 + j_base + 1][col] = -1
    V[(j_base + 1) * 7 + k_base][col] = 1
    V[(j_base + 1) * 7 + (k_base + 1)][col] = 1
    W[(k_base + 1) * 2 + 1][col] = 1
    col += 1
    
    # M4: a11(b21-b22)
    # a11 = A[0, j_base]
    # b21 = B[j_base+1, k_base], b22 = B[j_base+1, k_base+1]
    # U: A[0, j_base]
    # V: B[j_base+1, k_base] - B[j_base+1, k_base+1]
    # W: C11: +1, C22: -1
    U[j_base][col] = 1
    V[(j_base + 1) * 7 + k_base][col] = 1
    V[(j_base + 1) * 7 + (k_base + 1)][col] = -1
    W[k_base * 2 + 0][col] = 1
    W[(k_base + 1) * 2 + 1][col] = -1
    col += 1
    
    # M5: a22(b12-b11)
    # a22 = A[1, j_base+1]
    # b12 = B[j_base, k_base+1], b11 = B[j_base, k_base]
    # U: A[1, j_base+1]
    # V: B[j_base, k_base+1] - B[j_base, k_base]
    # W: C11: -1, C12: +1
    U[4 + j_base + 1][col] = 1
    V[j_base * 7 + (k_base + 1)][col] = 1
    V[j_base * 7 + k_base][col] = -1
    W[k_base * 2 + 0][col] = -1
    W[(k_base + 1) * 2 + 0][col] = 1
    col += 1
    
    # M6: (a11+a12)b22
    # a11 = A[0, j_base], a12 = A[0, j_base+1]
    # b22 = B[j_base+1, k_base+1]
    # U: A[0, j_base] + A[0, j_base+1]
    # V: B[j_base+1, k_base+1]
    # W: C12: +1, C22: -1
    U[j_base][col] = 1
    U[j_base + 1][col] = 1
    V[(j_base + 1) * 7 + (k_base + 1)][col] = 1
    W[(k_base + 1) * 2 + 0][col] = 1
    W[(k_base + 1) * 2 + 1][col] = -1
    col += 1
    
    # M7: (a21-a11)b11
    # a21 = A[1, j_base], a11 = A[0, j_base]
    # b11 = B[j_base, k_base]
    # U: A[1, j_base] - A[0, j_base]
    # V: B[j_base, k_base]
    # W: C11: +1, C21: -1
    U[4 + j_base][col] = 1
    U[j_base][col] = -1
    V[j_base * 7 + k_base][col] = 1
    W[k_base * 2 + 0][col] = 1
    W[k_base * 2 + 1][col] = -1
    col += 1
    
    return {"U": U, "V": V, "W": W}
