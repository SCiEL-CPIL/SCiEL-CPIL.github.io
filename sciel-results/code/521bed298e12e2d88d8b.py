"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 17 decomposition
    # 7 terms from Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # 4 terms for A[0:2, 2] * B[2, 0:2]
    # 6 terms for A[0:2, 0:3] * B[0:3, 2]
    
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # U[i*m + j, col] = u
    # V[j*p + k, col] = v
    # W[k*n + i, col] = w
    # This adds u * v * w to C[i, k]
    
    def add_term(i, j, k, col, u, v, w):
        U[i * m + j][col] += u
        V[j * p + k][col] += v
        W[k * n + i][col] += w

    col = 0
    
    # Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1]
    # B11=B[0,0], B12=B[0,1], B21=B[1,0], B22=B[1,1]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # M1
    # (A11+A22)(B11+B22)
    # Expand: A11B11 + A11B22 + A22B11 + A22B22
    # We need to add these to W
    # But Strassen uses linear combinations.
    # U[0,0] += 1, U[1,1] += 1
    # V[0,0] += 1, V[1,1] += 1
    # This gives (A11+A22)(B11+B22)
    # We need to distribute this to C[0,0], C[0,1], C[1,0], C[1,1]
    # C[0,0] += M1
    # C[0,1] += M1
    # C[1,0] += M1
    # C[1,1] += M1 ? No.
    # Strassen formulas:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 + M7
    
    # So for M1 (col 0):
    # U[0,0]=1, U[1,1]=1
    # V[0,0]=1, V[1,1]=1
    # W[0,0] += 1 (for C11)
    # W[1,0] += 0? No, W[k*n+i, col].
    # C[i,k] gets contribution from W[k*n+i, col].
    # M1 contributes to C[0,0], C[1,0]? No.
    # C[0,0] += M1 -> W[0*2+0, 0] += 1
    # C[0,1] += 0? No, M1 is not in C12.
    # C[1,0] += 0? No.
    # C[1,1] += M1 -> W[1*2+1, 0] += 1
    
    # Let's just add the terms directly to U, V, W.
    
    # M1
    U[0 * m + 0][col] += 1
    U[1 * m + 1][col] += 1
    V[0 * p + 0][col] += 1
    V[1 * p + 1][col] += 1
    W[0 * n + 0][col] += 1 # C[0,0]
    W[1 * n + 1][col] += 1 # C[1,1]
    col += 1
    
    # M2: (A21+A22)B11
    U[1 * m + 0][col] += 1
    U[1 * m + 1][col] += 1
    V[0 * p + 0][col] += 1
    W[0 * n + 1][col] += 1 # C[1,0]
    col += 1
    
    # M3: A11(B12-B22)
    U[0 * m + 0][col] += 1
    V[0 * p + 1][col] += 1
    V[1 * p + 1][col] -= 1
    W[1 * n + 0][col] += 1 # C[0,1]
    W[1 * n + 1][col] -= 1 # C[1,1]
    col += 1
    
    # M4: A22(B21-B11)
    U[1 * m + 1][col] += 1
    V[1 * p + 0][col] += 1
    V[0 * p + 0][col] -= 1
    W[0 * n + 0][col] -= 1 # C[0,0]
    W[0 * n + 1][col] += 1 # C[1,0]
    col += 1
    
    # M5: (A11+A12)B22
    U[0 * m + 0][col] += 1
    U[0 * m + 1][col] += 1
    V[1 * p + 1][col] += 1
    W[0 * n + 0][col] -= 1 # C[0,0]
    W[1 * n + 0][col] += 1 # C[0,1]
    col += 1
    
    # M6: (A21-A11)(B11+B12)
    U[1 * m + 0][col] += 1
    U[0 * m + 0][col] -= 1
    V[0 * p + 0][col] += 1
    V[0 * p + 1][col] += 1
    W[1 * n + 1][col] += 1 # C[1,1]
    col += 1
    
    # M7: (A12-A22)(B21+B22)
    U[0 * m + 1][col] += 1
    U[1 * m + 1][col] -= 1
    V[1 * p + 0][col] += 1
    V[1 * p + 1][col] += 1
    W[0 * n + 0][col] += 1 # C[0,0]
    W[1 * n + 1][col] += 1 # C[1,1]
    col += 1
    
    # Now the extra terms
    # T1: A13 B31 -> C[0,0]
    U[0 * m + 2][col] += 1
    V[2 * p + 0][col] += 1
    W[0 * n + 0][col] += 1
    col += 1
    
    # T2: A13 B32 -> C[0,1]
    U[0 * m + 2][col] += 1
    V[2 * p + 1][col] += 1
    W[1 * n + 0][col] += 1
    col += 1
    
    # T3: A23 B31 -> C[1,0]
    U[1 * m + 2][col] += 1
    V[2 * p + 0][col] += 1
    W[0 * n + 1][col] += 1
    col += 1
    
    # T4: A23 B32 -> C[1,1]
    U[1 * m + 2][col] += 1
    V[2 * p + 1][col] += 1
    W[1 * n + 1][col] += 1
    col += 1
    
    # T5: A11 B13 -> C[0,2]
    U[0 * m + 0][col] += 1
    V[0 * p + 2][col] += 1
    W[2 * n + 0][col] += 1
    col += 1
    
    # T6: A12 B23 -> C[0,2]
    U[0 * m + 1][col] += 1
    V[1 * p + 2][col] += 1
    W[2 * n + 0][col] += 1
    col += 1
    
    # T7: A13 B33 -> C[0,2]
    U[0 * m + 2][col] += 1
    V[2 * p + 2][col] += 1
    W[2 * n + 0][col] += 1
    col += 1
    
    # T8: A21 B13 -> C[1,2]
    U[1 * m + 0][col] += 1
    V[0 * p + 2][col] += 1
    W[2 * n + 1][col] += 1
    col += 1
    
    # T9: A22 B23 -> C[1,2]
    U[1 * m + 1][col] += 1
    V[1 * p + 2][col] += 1
    W[2 * n + 1][col] += 1
    col += 1
    
    # T10: A23 B33 -> C[1,2]
    U[1 * m + 2][col] += 1
    V[2 * p + 2][col] += 1
    W[2 * n + 1][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
