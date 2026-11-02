"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 terms for 2x2 multiplication
    # U indices: i*m + j (0:00, 1:01, 2:10, 3:11)
    # V indices: j*p + k
    # W indices: k*n + i
    
    # Define the 7 terms for a single block
    # Each term is a list of (U_idx, val), (V_idx_offset, val), (W_idx_offset, val)
    # V_idx_offset is relative to 2*q (for k=0) or 8+2*q (for k=1, j=1)
    # Actually, let's just compute global indices for V and W based on q.
    
    for q in range(4):
        base_k = 2 * q
        # V indices for B block q
        # B_00: j=0, k=base_k -> 0*8 + base_k
        # B_01: j=0, k=base_k+1 -> 0*8 + base_k + 1
        # B_10: j=1, k=base_k -> 1*8 + base_k
        # B_11: j=1, k=base_k+1 -> 1*8 + base_k + 1
        
        v00 = base_k
        v01 = base_k + 1
        v10 = 8 + base_k
        v11 = 8 + base_k + 1
        
        # W indices for C block q
        # C_00: k=base_k, i=0 -> base_k*2 + 0
        # C_01: k=base_k, i=1 -> base_k*2 + 1
        # C_10: k=base_k+1, i=0 -> (base_k+1)*2 + 0
        # C_11: k=base_k+1, i=1 -> (base_k+1)*2 + 1
        
        w00 = base_k * 2
        w01 = base_k * 2 + 1
        w10 = (base_k + 1) * 2
        w11 = (base_k + 1) * 2 + 1
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        r = 7 * q + 0
        U[0][r] = 1
        U[3][r] = 1
        V[v00][r] = 1
        V[v11][r] = 1
        W[w00][r] = 1
        W[w11][r] = 1
        
        # Term 2: M2 = (A10+A11)B00
        r = 7 * q + 1
        U[2][r] = 1
        U[3][r] = 1
        V[v00][r] = 1
        W[w10][r] = 1
        W[w11][r] = -1
        
        # Term 3: M3 = A00(B01-B11)
        r = 7 * q + 2
        U[0][r] = 1
        V[v01][r] = 1
        V[v11][r] = -1
        W[w01][r] = 1
        W[w11][r] = 1
        
        # Term 4: M4 = A11(B10-B00)
        r = 7 * q + 3
        U[3][r] = 1
        V[v10][r] = 1
        V[v00][r] = -1
        W[w00][r] = 1
        W[w10][r] = 1
        
        # Term 5: M5 = (A00+A01)B11
        r = 7 * q + 4
        U[0][r] = 1
        U[1][r] = 1
        V[v11][r] = 1
        W[w00][r] = -1
        W[w11][r] = -1
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        r = 7 * q + 5
        U[2][r] = 1
        U[0][r] = -1
        V[v00][r] = 1
        V[v01][r] = 1
        W[w11][r] = 1
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        r = 7 * q + 6
        U[1][r] = 1
        U[3][r] = -1
        V[v10][r] = 1
        V[v11][r] = 1
        W[w00][r] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
