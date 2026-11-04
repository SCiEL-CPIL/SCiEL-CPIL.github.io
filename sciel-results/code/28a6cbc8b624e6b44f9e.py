import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # M1: (X00+X11)(Y00+Y11)
        # U: (i, j) + (i+1, j+1)
        U[i_off * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        # V: (j, k) + (j+1, k+1)
        V[j_off * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        # W: (k, i) + (k+1, i+1)
        W[k_off * n + i_off][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1
        
        # M2: (X10+X11)Y00
        U[i_off * m + j_off][col] = 0 # X10 is i+1, j+0
        U[(i_off+1) * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = -1
        col += 1
        
        # M3: X00(Y10-Y11)
        U[i_off * m + j_off][col] = 1
        V[(j_off+1) * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = -1
        W[k_off * n + i_off][col] = 1
        W[(k_off+1) * n + i_off][col] = 1
        col += 1
        
        # M4: X11(Y01-Y00)
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        V[j_off * p + k_off][col] = -1
        W[k_off * n + (i_off+1)][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1
        
        # M5: (X00+X10)Y11
        U[i_off * m + j_off][col] = 1
        U[(i_off+1) * m + j_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = -1
        W[(k_off+1) * n + i_off][col] = 1
        col += 1
        
        # M6: (X11-X00)(Y00+Y10)
        U[(i_off+1) * m + (j_off+1)][col] = 1
        U[i_off * m + j_off][col] = -1
        V[j_off * p + k_off][col] = 1
        V[(j_off+1) * p + k_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1
        
        # M7: (X01-X11)(Y11+Y10)
        U[i_off * m + (j_off+1)][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = -1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        V[(j_off+1) * p + k_off][col] = 1
        W[k_off * n + i_off][col] = 1
        W[(k_off+1) * n + i_off][col] = -1
        col += 1

    # Helper to add a 1x2x2 block (4 scalar mults)
    def add_1x2x2(i_idx, j_off, k_off):
        nonlocal col
        # A[i, j] * B[j, k] -> C[i, k]
        
        # Term 1: A[i, j_off] * B[j_off, k_off]
        U[i_idx * m + j_off][col] = 1
        V[j_off * p + k_off][col] = 1
        W[k_off * n + i_idx][col] = 1
        col += 1
        
        # Term 2: A[i, j_off] * B[j_off, k_off+1]
        U[i_idx * m + j_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        W[(k_off+1) * n + i_idx][col] = 1
        col += 1
        
        # Term 3: A[i, j_off+1] * B[j_off+1, k_off]
        U[i_idx * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + k_off][col] = 1
        W[k_off * n + i_idx][col] = 1
        col += 1
        
        # Term 4: A[i, j_off+1] * B[j_off+1, k_off+1]
        U[i_idx * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[(k_off+1) * n + i_idx][col] = 1
        col += 1

    # Top 2 rows (i=0,1)
    add_strassen(0, 0, 0) # A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen(0, 2, 0) # A[0:2, 2:4] * B[2:4, 0:2]
    add_strassen(0, 0, 2) # A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen(0, 2, 2) # A[0:2, 2:4] * B[2:4, 2:4]
    
    # Bottom 1 row (i=2)
    add_1x2x2(2, 0, 0) # A[2, 0:2] * B[0:2, 0:2]
    add_1x2x2(2, 2, 0) # A[2, 2:4] * B[2:4, 0:2]
    add_1x2x2(2, 0, 2) # A[2, 0:2] * B[0:2, 2:4]
    add_1x2x2(2, 2, 2) # A[2, 2:4] * B[2:4, 2:4]
    
    return {"U": U, "V": V, "W": W}
