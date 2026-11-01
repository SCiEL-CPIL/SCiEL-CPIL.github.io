"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK. Rank 49 decomposition."""
DIMENSIONS = [2, 2, 14]

import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    # Use Strassen decomposition: 7 blocks of 2x2 * 2x2, each rank 7
    # Total rank = 7 * 7 = 49
    rank = 49
    U = [[0] * rank for _ in range(n * m)]   # shape (4, 49)
    V = [[0] * rank for _ in range(m * p)]   # shape (28, 49)
    W = [[0] * rank for _ in range(p * n)]   # shape (28, 49)
    
    # A indices: u = i*m + j = 2*i + j
    # B indices: v = j*p + k = 14*j + k
    # C indices: w = k*n + i = 2*k + i
    
    # For each block q (q=0..6), covering k0=2q, k1=2q+1:
    # Strassen products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # Results:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    for q in range(7):
        k0 = 2 * q
        k1 = 2 * q + 1
        col_base = q * 7
        
        # B row indices
        v_B00 = 14 * 0 + k0  # B[0][k0]
        v_B01 = 14 * 0 + k1  # B[0][k1]
        v_B10 = 14 * 1 + k0  # B[1][k0]
        v_B11 = 14 * 1 + k1  # B[1][k1]
        
        # C column indices (w = k*n + i = 2*k + i)
        w_C00 = 2 * k0 + 0   # C[0][k0]
        w_C01 = 2 * k1 + 0   # C[0][k1]
        w_C10 = 2 * k0 + 1   # C[1][k0]
        w_C11 = 2 * k1 + 1   # C[1][k1]
        
        # A indices
        u_A00 = 2 * 0 + 0    # A[0][0]
        u_A01 = 2 * 0 + 1    # A[0][1]
        u_A10 = 2 * 1 + 0    # A[1][0]
        u_A11 = 2 * 1 + 1    # A[1][1]
        
        # M1 = (A00 + A11)(B00 + B11)
        c = col_base + 0
        U[u_A00][c] += 1
        U[u_A11][c] += 1
        V[v_B00][c] += 1
        V[v_B11][c] += 1
        W[w_C00][c] += 1
        W[w_C11][c] += 1
        
        # M2 = (A10 + A11)B00
        c = col_base + 1
        U[u_A10][c] += 1
        U[u_A11][c] += 1
        V[v_B00][c] += 1
        W[w_C10][c] += 1
        W[w_C11][c] -= 1
        
        # M3 = A00(B01 - B11)
        c = col_base + 2
        U[u_A00][c] += 1
        V[v_B01][c] += 1
        V[v_B11][c] -= 1
        W[w_C01][c] += 1
        W[w_C11][c] += 1
        
        # M4 = A11(B10 - B00)
        c = col_base + 3
        U[u_A11][c] += 1
        V[v_B10][c] += 1
        V[v_B00][c] -= 1
        W[w_C00][c] += 1
        W[w_C10][c] += 1
        
        # M5 = (A00 + A01)B11
        c = col_base + 4
        U[u_A00][c] += 1
        U[u_A01][c] += 1
        V[v_B11][c] += 1
        W[w_C00][c] -= 1
        W[w_C01][c] += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        c = col_base + 5
        U[u_A10][c] += 1
        U[u_A00][c] -= 1
        V[v_B00][c] += 1
        V[v_B01][c] += 1
        W[w_C11][c] += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        c = col_base + 6
        U[u_A01][c] += 1
        U[u_A11][c] -= 1
        V[v_B10][c] += 1
        V[v_B11][c] += 1
        W[w_C00][c] += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
