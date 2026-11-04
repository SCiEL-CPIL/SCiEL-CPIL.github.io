"""Matrix multiplication 3x4x8 via split-and-combine with Strassen."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    total_rank = 88
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 88
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 88
    W = [[0] * total_rank for _ in range(p * n)]  # 24 x 88
    
    col = 0
    
    def strassen_block(i0, j0, k0):
        """Strassen for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]"""
        nonlocal col
        A00 = (i0, j0)
        A01 = (i0, j0+1)
        A10 = (i0+1, j0)
        A11 = (i0+1, j0+1)
        B00 = (j0, k0)
        B01 = (j0, k0+1)
        B10 = (j0+1, k0)
        B11 = (j0+1, k0+1)
        C00 = (k0, i0)
        C01 = (k0, i0+1)
        C10 = (k0+1, i0)
        C11 = (k0+1, i0+1)
        
        # M1 = (A00+A11)(B00+B11)
        U[A00[0]*m + A00[1]][col] = 1
        U[A11[0]*m + A11[1]][col] = 1
        V[B00[0]*p + B00[1]][col] = 1
        V[B11[0]*p + B11[1]][col] = 1
        W[C00[0]*n + C00[1]][col] = 1
        W[C01[0]*n + C01[1]][col] = 1
        W[C10[0]*n + C10[1]][col] = 1
        W[C11[0]*n + C11[1]][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[A10[0]*m + A10[1]][col] = 1
        U[A11[0]*m + A11[1]][col] = 1
        V[B00[0]*p + B00[1]][col] = 1
        W[C10[0]*n + C10[1]][col] = 1
        W[C11[0]*n + C11[1]][col] = 1
        col += 1
        
        # M3 = A00(B01-B11)
        U[A00[0]*m + A00[1]][col] = 1
        V[B01[0]*p + B01[1]][col] = 1
        V[B11[0]*p + B11[1]][col] = -1
        W[C01[0]*n + C01[1]][col] = 1
        W[C11[0]*n + C11[1]][col] = 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[A11[0]*m + A11[1]][col] = 1
        V[B10[0]*p + B10[1]][col] = 1
        V[B00[0]*p + B00[1]][col] = -1
        W[C00[0]*n + C00[1]][col] = 1
        W[C10[0]*n + C10[1]][col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[A00[0]*m + A00[1]][col] = 1
        U[A01[0]*m + A01[1]][col] = 1
        V[B11[0]*p + B11[1]][col] = 1
        W[C00[0]*n + C00[1]][col] = -1
        W[C01[0]*n + C01[1]][col] = 1
        W[C11[0]*n + C11[1]][col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[A10[0]*m + A10[1]][col] = 1
        U[A00[0]*m + A00[1]][col] = -1
        V[B00[0]*p + B00[1]][col] = 1
        V[B01[0]*p + B01[1]][col] = 1
        W[C11[0]*n + C11[1]][col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[A01[0]*m + A01[1]][col] = 1
        U[A11[0]*m + A11[1]][col] = -1
        V[B10[0]*p + B10[1]][col] = 1
        V[B11[0]*p + B11[1]][col] = 1
        W[C00[0]*n + C00[1]][col] = 1
        W[C11[0]*n + C11[1]][col] = 1
        col += 1
    
    def schoolbook_1x2x2(i0, j0, k0):
        """Schoolbook for A[i0, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0, k0:k0+2]"""
        nonlocal col
        U[i0*m + j0][col] = 1
        V[j0*p + k0][col] = 1
        W[k0*n + i0][col] = 1
        col += 1
        
        U[i0*m + j0][col] = 1
        V[j0*p + k0+1][col] = 1
        W[(k0+1)*n + i0][col] = 1
        col += 1
        
        U[i0*m + j0+1][col] = 1
        V[(j0+1)*p + k0][col] = 1
        W[k0*n + i0][col] = 1
        col += 1
        
        U[i0*m + j0+1][col] = 1
        V[(j0+1)*p + k0+1][col] = 1
        W[(k0+1)*n + i0][col] = 1
        col += 1
    
    # Top-left: C[0:2, 0:4]
    strassen_block(0, 0, 0)
    strassen_block(0, 2, 0)
    strassen_block(0, 0, 2)
    strassen_block(0, 2, 2)
    
    # Top-right: C[0:2, 4:8]
    strassen_block(0, 0, 4)
    strassen_block(0, 2, 4)
    strassen_block(0, 0, 6)
    strassen_block(0, 2, 6)
    
    # Bottom-left: C[2, 0:4]
    schoolbook_1x2x2(2, 0, 0)
    schoolbook_1x2x2(2, 2, 0)
    schoolbook_1x2x2(2, 0, 2)
    schoolbook_1x2x2(2, 2, 2)
    
    # Bottom-right: C[2, 4:8]
    schoolbook_1x2x2(2, 0, 4)
    schoolbook_1x2x2(2, 2, 4)
    schoolbook_1x2x2(2, 0, 6)
    schoolbook_1x2x2(2, 2, 6)
    
    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
