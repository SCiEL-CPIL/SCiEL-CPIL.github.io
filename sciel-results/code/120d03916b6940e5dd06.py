"""Matrix multiplication 3x4x5 tensor decomposition."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 5
    
    # We'll build U, V, W as lists of lists
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    # Helper to add a schoolbook rank-1 term
    def add_schoolbook(i, j, k, col):
        U[i*m + j][col] = 1
        V[j*p + k][col] = 1
        W[k*n + i][col] = 1
    
    # Helper to add Strassen 2x2x2 block
    # For indices (i0,i1) x (j0,j1) x (k0,k1)
    # Strassen: 7 products
    # M1 = (A00+A11)(B01+B10)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    def add_strassen(i0, i1, j0, j1, k0, k1, base_col):
        # 7 terms for Strassen
        # Term 1: (A[i0,j0]+A[i1,j1]) * (B[j0,k1]+B[j1,k0])
        U[i0*m+j0][base_col+0] = 1
        U[i1*m+j1][base_col+0] = 1
        V[j0*p+k1][base_col+0] = 1
        V[j1*p+k0][base_col+0] = 1
        W[k0*n+i0][base_col+0] = 1  # C00 += M1
        W[k1*n+i0][base_col+0] = 1  # C01 += M1? No, let me redo
        
        # Actually, let me think about this more carefully.
        # The output C[i,k] = sum_j A[i,j]*B[j,k]
        # For the 2x2x2 block with rows {i0,i1}, cols {k0,k1}, mid {j0,j1}
        # C[i0,k0] = A[i0,j0]*B[j0,k0] + A[i0,j1]*B[j1,k0]
        # C[i0,k1] = A[i0,j0]*B[j0,k1] + A[i0,j1]*B[j1,k1]
        # C[i1,k0] = A[i1,j0]*B[j0,k0] + A[i1,j1]*B[j1,k0]
        # C[i1,k1] = A[i1,j0]*B[j0,k1] + A[i1,j1]*B[j1,k1]
        
        # Strassen formulas:
        # M1 = (A00+A11)(B01+B10) -> contributes to C00, C11
        # M2 = (A10+A11)B00 -> contributes to C10, C11
        # M3 = A00(B10-B00) -> contributes to C00, C01
        # M4 = A11(B01-B11) -> contributes to C00, C11
        # M5 = (A00+A01)B11 -> contributes to C00, C01
        # M6 = (A10-A00)(B00+B10) -> contributes to C10, C11
        # M7 = (A01-A11)(B01+B11) -> contributes to C00, C01
        
        # Let me use standard Strassen with proper signs:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1 = (A00+A11)(B01+B10)
        U[i0*m+j0][base_col+0] = 1
        U[i1*m+j1][base_col+0] = 1
        V[j0*p+k1][base_col+0] = 1
        V[j1*p+k0][base_col+0] = 1
        W[k0*n+i0][base_col+0] = 1
        W[k1*n+i1][base_col+0] = 1
        
        # M2 = (A10+A11)B00
        U[i1*m+j0][base_col+1] = 1
        U[i1*m+j1][base_col+1] = 1
        V[j0*p+k0][base_col+1] = 1
        W[k0*n+i1][base_col+1] = 1
        W[k1*n+i1][base_col+1] = -1
        
        # M3 = A00(B10-B00)
        U[i0*m+j0][base_col+2] = 1
        V[j1*p+k0][base_col+2] = 1
        V[j0*p+k0][base_col+2] = -1
        W[k0*n+i0][base_col+2] = 1
        W[k1*n+i0][base_col+2] = 1
        
        # M4 = A11(B01-B11)
        U[i1*m+j1][base_col+3] = 1
        V[j0*p+k1][base_col+3] = 1
        V[j1*p+k1][base_col+3] = -1
        W[k0*n+i0][base_col+3] = 1
        W[k1*n+i1][base_col+3] = 1
        
        # M5 = (A00+A01)B11
        U[i0*m+j0][base_col+4] = 1
        U[i0*m+j1][base_col+4] = 1
        V[j1*p+k1][base_col+4] = 1
        W[k0*n+i0][base_col+4] = -1
        W[k1*n+i0][base_col+4] = 1
        
        # M6 = (A10-A00)(B00+B10)
        U[i1*m+j0][base_col+5] = 1
        U[i0*m+j0][base_col+5] = -1
        V[j0*p+k0][base_col+5] = 1
        V[j1*p+k0][base_col+5] = 1
        W[k0*n+i1][base_col+5] = 1
        W[k1*n+i1][base_col+5] = 1
        
        # M7 = (A01-A11)(B01+B11)
        U[i0*m+j1][base_col+6] = 1
        U[i1*m+j1][base_col+6] = -1
        V[j0*p+k1][base_col+6] = 1
        V[j1*p+k1][base_col+6] = 1
        W[k0*n+i0][base_col+6] = 1
        W[k1*n+i0][base_col+6] = 1
        
        return base_col + 7
    
    # Initialize U, V, W with zeros
    # We'll figure out total rank first
    # 4 Strassen blocks (2x2x2) = 4*7 = 28
    # 2x4x1 block (i=0,1; j=0..3; k=4) = 8
    # 1x4x5 block (i=2; j=0..3; k=0..4) = 20
    # Total = 56
    
    total_rank = 56
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col = 0
    # Strassen block 1: i={0,1}, j={0,1}, k={0,1}
    col = add_strassen(0, 1, 0, 1, 0, 1, col)
    # Strassen block 2: i={0,1}, j={2,3}, k={2,3}
    col = add_strassen(0, 1, 2, 3, 2, 3, col)
    # Strassen block 3: i={0,1}, j={0,1}, k={2,3}
    col = add_strassen(0, 1, 0, 1, 2, 3, col)
    # Strassen block 4: i={0,1}, j={2,3}, k={0,1}
    col = add_strassen(0, 1, 2, 3, 0, 1, col)
    
    # Schoolbook: i={0,1}, j={0,1,2,3}, k={4}
    for j in range(4):
        for i in range(2):
            add_schoolbook(i, j, 4, col)
            col += 1
    
    # Schoolbook: i={2}, j={0,1,2,3}, k={0,1,2,3,4}
    for j in range(4):
        for k in range(5):
            add_schoolbook(2, j, k, col)
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
