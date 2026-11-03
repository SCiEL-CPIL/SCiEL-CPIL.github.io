"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 * B1 using Strassen for 2x2x2 blocks
    # A1 is 2x2 (rows 0-1, cols 0-1 of A)
    # B1 is 2x8 (rows 0-1, cols 0-7 of B)
    # Split B1 into 4 blocks of 2x2
    
    # Indices for A1 in U
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 2
    # i=1, j=1 -> 3
    u_a11 = 0
    u_a12 = 1
    u_a21 = 2
    u_a22 = 3
    
    for b in range(4):
        # B block b corresponds to k in [2b, 2b+1]
        # V indices for B1 block b:
        # j=0, k=2b -> 2b
        # j=0, k=2b+1 -> 2b+1
        # j=1, k=2b -> 8+2b
        # j=1, k=2b+1 -> 8+2b+1
        v_b11 = 2 * b
        v_b12 = 2 * b + 1
        v_b21 = 8 + 2 * b
        v_b22 = 8 + 2 * b + 1
        
        # W indices for C1 block b:
        # i=0, k=2b -> 2b*2 + 0 = 4b
        # i=0, k=2b+1 -> (2b+1)*2 + 0 = 4b+2
        # i=1, k=2b -> 2b*2 + 1 = 4b+1
        # i=1, k=2b+1 -> (2b+1)*2 + 1 = 4b+3
        w_c11 = 4 * b
        w_c12 = 4 * b + 2
        w_c21 = 4 * b + 1
        w_c22 = 4 * b + 3
        
        # Strassen steps
        # m1 = (A11 + A12) * (B21 + B22)
        U[u_a11][col] = 1
        U[u_a12][col] = 1
        V[v_b21][col] = 1
        V[v_b22][col] = 1
        W[w_c11][col] = 1
        W[w_c22][col] = 1
        col += 1
        
        # m2 = (A11 + A12) * B22
        U[u_a11][col] = 1
        U[u_a12][col] = 1
        V[v_b22][col] = 1
        W[w_c12][col] = 1
        W[w_c22][col] = -1
        col += 1
        
        # m3 = A11 * (B11 - B21)
        U[u_a11][col] = 1
        V[v_b11][col] = 1
        V[v_b21][col] = -1
        W[w_c21][col] = 1
        W[w_c22][col] = 1
        col += 1
        
        # m4 = A22 * (B22 - B12)
        U[u_a22][col] = 1
        V[v_b22][col] = 1
        V[v_b12][col] = -1
        W[w_c11][col] = 1
        W[w_c22][col] = 1
        col += 1
        
        # m5 = (A22 + A21) * B11
        U[u_a22][col] = 1
        U[u_a21][col] = 1
        V[v_b11][col] = 1
        W[w_c11][col] = -1
        W[w_c12][col] = 1
        col += 1
        
        # m6 = (A22 + A21) * (B12 - B22)
        U[u_a22][col] = 1
        U[u_a21][col] = 1
        V[v_b12][col] = 1
        V[v_b22][col] = -1
        W[w_c21][col] = 1
        col += 1
        
        # m7 = (A11 - A21) * (B11 + B12)
        U[u_a11][col] = 1
        U[u_a21][col] = -1
        V[v_b11][col] = 1
        V[v_b12][col] = 1
        W[w_c11][col] = 1
        col += 1

    # Part 2: A2 * B2 using Schoolbook for 2x1x8
    # A2 is 2x1 (rows 0-1, col 2 of A)
    # B2 is 1x8 (row 2, cols 0-7 of B)
    
    # U indices for A2
    u_a21 = 4 # i=0, j=2
    u_a22 = 5 # i=1, j=2
    
    for k in range(8):
        # V index for B2, k
        v_b2k = 16 + k # j=2, k
        
        # W indices for C2, k
        w_c1k = 2 * k # i=0, k
        w_c2k = 2 * k + 1 # i=1, k
        
        # C[0,k] = A[0,2] * B[2,k]
        U[u_a21][col] = 1
        V[v_b2k][col] = 1
        W[w_c1k][col] = 1
        col += 1
        
        # C[1,k] = A[1,2] * B[2,k]
        U[u_a22][col] = 1
        V[v_b2k][col] = 1
        W[w_c2k][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
