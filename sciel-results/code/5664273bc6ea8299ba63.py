"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    # Rank 49 using 7 Strassen blocks of 2x2x2
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for A (2x2):
    # 0: A[0,0]
    # 1: A[0,1]
    # 2: A[1,0]
    # 3: A[1,1]
    
    # For each block b (0 to 6), corresponding to B columns 2b, 2b+1
    for b in range(7):
        # B indices for this block:
        # e = B[0, 2b] -> index 0*14 + 2b
        # f = B[0, 2b+1] -> index 0*14 + 2b + 1
        # g = B[1, 2b] -> index 1*14 + 2b
        # h = B[1, 2b+1] -> index 1*14 + 2b + 1
        
        idx_e = 0 * 14 + 2 * b
        idx_f = 0 * 14 + 2 * b + 1
        idx_g = 1 * 14 + 2 * b
        idx_h = 1 * 14 + 2 * b + 1
        
        # C indices for this block (W index k*n + i, n=2):
        # C[0, 2b] -> k=2b, i=0 -> 2b*2 + 0
        # C[0, 2b+1] -> k=2b+1, i=0 -> (2b+1)*2 + 0
        # C[1, 2b] -> k=2b, i=1 -> 2b*2 + 1
        # C[1, 2b+1] -> k=2b+1, i=1 -> (2b+1)*2 + 1
        
        idx_c00 = 2 * b * 2 + 0
        idx_c01 = (2 * b + 1) * 2 + 0
        idx_c10 = 2 * b * 2 + 1
        idx_c11 = (2 * b + 1) * 2 + 1
        
        # Strassen products P1..P7
        
        # P1 = (A00 + A11) * (e + h)
        col = 7 * b + 0
        U[0][col] = 1  # A00
        U[3][col] = 1  # A11
        V[idx_e][col] = 1
        V[idx_h][col] = 1
        W[idx_c00][col] = 1
        W[idx_c11][col] = 1
        
        # P2 = (A10 + A11) * e
        col = 7 * b + 1
        U[2][col] = 1  # A10
        U[3][col] = 1  # A11
        V[idx_e][col] = 1
        W[idx_c10][col] = 1
        W[idx_c11][col] = -1
        
        # P3 = A00 * (f - h)
        col = 7 * b + 2
        U[0][col] = 1  # A00
        V[idx_f][col] = 1
        V[idx_h][col] = -1
        W[idx_c01][col] = 1
        W[idx_c11][col] = 1
        
        # P4 = A11 * (g - e)
        col = 7 * b + 3
        U[3][col] = 1  # A11
        V[idx_g][col] = 1
        V[idx_e][col] = -1
        W[idx_c00][col] = 1
        W[idx_c10][col] = 1
        
        # P5 = (A00 + A01) * h
        col = 7 * b + 4
        U[0][col] = 1  # A00
        U[1][col] = 1  # A01
        V[idx_h][col] = 1
        W[idx_c00][col] = -1
        W[idx_c01][col] = 1
        
        # P6 = (A10 - A00) * (e + f)
        col = 7 * b + 5
        U[2][col] = 1  # A10
        U[0][col] = -1 # A00
        V[idx_e][col] = 1
        V[idx_f][col] = 1
        W[idx_c11][col] = 1
        
        # P7 = (A01 - A11) * (g + h)
        col = 7 * b + 6
        U[1][col] = 1  # A01
        U[3][col] = -1 # A11
        V[idx_g][col] = 1
        V[idx_h][col] = 1
        W[idx_c00][col] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
