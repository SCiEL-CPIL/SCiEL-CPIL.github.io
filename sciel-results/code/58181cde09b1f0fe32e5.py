"""Rewritten program for matrix multiplication 2x2 by 2x5 tensor decomposition."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 18 decomposition:
    # 2 Strassen blocks for first 4 columns (rank 7 each)
    # 1 Schoolbook block for last column (rank 4)
    rank = 18
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen block for B columns k_start to k_start+1
    # and C columns k_start to k_start+1
    # t_offset is the starting column index in U, V, W
    def add_strassen_block(t_offset, k_start):
        # Strassen products M1..M7
        # U (A) coefficients:
        # M1: A00+A11
        U[0][t_offset+0] = 1
        U[3][t_offset+0] = 1
        # M2: A10+A11
        U[2][t_offset+1] = 1
        U[3][t_offset+1] = 1
        # M3: A00
        U[0][t_offset+2] = 1
        # M4: A11
        U[3][t_offset+3] = 1
        # M5: A00+A01
        U[0][t_offset+4] = 1
        U[1][t_offset+4] = 1
        # M6: A10-A00
        U[2][t_offset+5] = 1
        U[0][t_offset+5] = -1
        # M7: A01-A11
        U[1][t_offset+6] = 1
        U[3][t_offset+6] = -1
        
        # V (B) coefficients for columns k_start, k_start+1
        # Indices: j*p + k
        k0 = k_start
        k1 = k_start + 1
        # M1: B00+B11
        V[0*p + k0][t_offset+0] = 1
        V[1*p + k1][t_offset+0] = 1
        # M2: B00
        V[0*p + k0][t_offset+1] = 1
        # M3: B01-B11
        V[0*p + k1][t_offset+2] = 1
        V[1*p + k1][t_offset+2] = -1
        # M4: B10-B00
        V[1*p + k0][t_offset+3] = 1
        V[0*p + k0][t_offset+3] = -1
        # M5: B11
        V[1*p + k1][t_offset+4] = 1
        # M6: B00+B01
        V[0*p + k0][t_offset+5] = 1
        V[0*p + k1][t_offset+5] = 1
        # M7: B10+B11
        V[1*p + k0][t_offset+6] = 1
        V[1*p + k1][t_offset+6] = 1
        
        # W (C) coefficients for columns k_start, k_start+1
        # Indices: k*n + i
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1
        W[k0*n + 0][t_offset+0] = 1
        W[k1*n + 1][t_offset+0] = 1
        # M2
        W[k0*n + 1][t_offset+1] = 1
        W[k1*n + 1][t_offset+1] = -1
        # M3
        W[k1*n + 0][t_offset+2] = 1
        W[k1*n + 1][t_offset+2] = 1
        # M4
        W[k0*n + 0][t_offset+3] = 1
        W[k0*n + 1][t_offset+3] = 1
        # M5
        W[k0*n + 0][t_offset+4] = -1
        W[k1*n + 0][t_offset+4] = 1
        # M6
        W[k1*n + 1][t_offset+5] = 1
        # M7
        W[k0*n + 0][t_offset+6] = 1

    # Add first Strassen block for B cols 0,1
    add_strassen_block(0, 0)
    
    # Add second Strassen block for B cols 2,3
    add_strassen_block(7, 2)
    
    # Add schoolbook for B col 4
    t = 14
    # C04 = A00*B04 + A01*B14
    U[0][t] = 1
    V[0*p + 4][t] = 1
    W[4*n + 0][t] = 1
    t += 1
    
    U[1][t] = 1
    V[1*p + 4][t] = 1
    W[4*n + 0][t] = 1
    t += 1
    
    # C14 = A10*B04 + A11*B14
    U[2][t] = 1
    V[0*p + 4][t] = 1
    W[4*n + 1][t] = 1
    t += 1
    
    U[3][t] = 1
    V[1*p + 4][t] = 1
    W[4*n + 1][t] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
