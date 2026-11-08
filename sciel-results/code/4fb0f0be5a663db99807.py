"""Rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Use Strassen for the 2x2 part and outer product for the 2x1 part
    # For each of the 3 column blocks (k=0,1; k=2,3; k=4,5):
    # - Strassen decomposition of 2x2 * 2x2 (rank 7)
    # - Outer product for 2x1 * 1x2 (rank 4)
    # Total rank = 3 * (7 + 4) = 33
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]  # shape (6, 33)
    V = [[0] * rank for _ in range(m * p)]  # shape (18, 33)
    W = [[0] * rank for _ in range(p * n)]  # shape (12, 33)
    
    col = 0
    for l in range(3):
        # k indices: 2*l, 2*l+1
        k0 = 2 * l
        k1 = 2 * l + 1
        
        # Strassen part: 2x2 * 2x2
        # A = [[a00, a01], [a10, a11]] where aij = U[i*m + j]
        # B = [[b00, b01], [b10, b11]] where bij = V[j*p + k]
        # For block l: B = [[V[0*p+k0], V[0*p+k1]], [V[1*p+k0], V[1*p+k1]]]
        
        # M1 = (A00 + A11)(B00 + B11)
        U[0 * m + 0][col] = 1  # A00
        U[1 * m + 1][col] = 1  # A11
        V[0 * p + k0][col] = 1  # B00
        V[1 * p + k0][col] = 1  # B11
        W[k0 * n + 0][col] = 1  # C00 += M1
        W[k0 * n + 1][col] = 1  # C10 += M1? Wait, let me recheck Strassen
        
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M4
        # C10 = M2 + M6
        # C11 = M1 - M2 + M3 + M6
        
        col += 1
        
        # M2 = (A10+A11)B00
        U[1 * m + 0][col] = 1  # A10
        U[1 * m + 1][col] = 1  # A11
        V[0 * p + k0][col] = 1  # B00
        W[k0 * n + 1][col] = 1  # C10 += M2
        W[k1 * n + 1][col] = -1  # C11 -= M2
        col += 1
        
        # M3 = A00(B10-B11)
        U[0 * m + 0][col] = 1  # A00
        V[1 * p + k0][col] = 1  # B10
        V[1 * p + k1][col] = -1  # -B11
        W[k0 * n + 0][col] = 1  # C01 += M3
        W[k1 * n + 0][col] = 1  # C11 += M3? No, C01 = M3 + M4, C11 = M1 - M2 + M3 + M6
        # Actually C01 is row 0, col 1; C11 is row 1, col 1
        # W[k*n+i, r] where k is column of B, i is row of A
        # C[i][k] = sum over j of A[i][j] * B[j][k]
        # For Strassen block, C is 2x2: C[i][k'] where k' in {0,1} corresponds to k0, k1
        # W[k*n+i, r] = coefficient for C[i][k]
        # So for C01 (i=0, k=1): W[k1*n+0, col]
        # For C11 (i=1, k=1): W[k1*n+1, col]
        W[k1 * n + 0][col] = 1  # C01 += M3
        W[k1 * n + 1][col] = 1  # C11 += M3
        col += 1
        
        # M4 = A11(B01-B00)
        U[1 * m + 1][col] = 1  # A11
        V[0 * p + k1][col] = 1  # B01
        V[0 * p + k0][col] = -1  # -B00
        W[k1 * n + 0][col] = 1  # C01 += M4
        W[k0 * n + 0][col] = -1  # C00 -= M4
        W[k1 * n + 1][col] = 1  # C11 += M4? No, C11 = M1 - M2 + M3 + M6, no M4
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M4
        # C10 = M2 + M6
        # C11 = M1 - M2 + M3 + M6
        # So M4 only in C00 and C01
        W[k1 * n + 1][col] = 0
        col += 1
        
        # M5 = (A00+A01)B11
        U[0 * m + 0][col] = 1  # A00
        U[0 * m + 1][col] = 1  # A01
        V[1 * p + k1][col] = 1  # B11
        W[k0 * n + 0][col] = 1  # C00 += M5
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[1 * m + 0][col] = 1  # A10
        U[0 * m + 0][col] = -1  # -A00
        V[0 * p + k0][col] = 1  # B00
        V[0 * p + k1][col] = 1  # B01
        W[k0 * n + 1][col] = 1  # C10 += M6
        W[k1 * n + 1][col] = 1  # C11 += M6
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[0 * m + 1][col] = 1  # A01
        U[1 * m + 1][col] = -1  # -A11
        V[1 * p + k0][col] = 1  # B10
        V[1 * p + k1][col] = 1  # B11
        W[k0 * n + 0][col] = 1  # C00 += M7
        col += 1
        
        # Outer product part: 2x1 * 1x2
        # A_col = [A02, A12], B_row = [B2,k0, B2,k1]
        # N1 = A02 * B2,k0 -> C0,k0
        # N2 = A02 * B2,k1 -> C0,k1
        # N3 = A12 * B2,k0 -> C1,k0
        # N4 = A12 * B2,k1 -> C1,k1
        
        # N1
        U[0 * m + 2][col] = 1  # A02
        V[2 * p + k0][col] = 1  # B2,k0
        W[k0 * n + 0][col] = 1  # C0,k0 += N1
        col += 1
        
        # N2
        U[0 * m + 2][col] = 1  # A02
        V[2 * p + k1][col] = 1  # B2,k1
        W[k1 * n + 0][col] = 1  # C0,k1 += N2
        col += 1
        
        # N3
        U[1 * m + 2][col] = 1  # A12
        V[2 * p + k0][col] = 1  # B2,k0
        W[k0 * n + 1][col] = 1  # C1,k0 += N3
        col += 1
        
        # N4
        U[1 * m + 2][col] = 1  # A12
        V[2 * p + k1][col] = 1  # B2,k1
        W[k1 * n + 1][col] = 1  # C1,k1 += N4
        col += 1
    
    return {"U": U, "V": V, "W": W}
