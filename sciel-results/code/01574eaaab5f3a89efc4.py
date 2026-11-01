"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 21 decomposition using Strassen's algorithm for 2x2x2 blocks
    # p=6 implies 3 blocks of size 2
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(num_blocks):
        # Block b covers columns 2b and 2b+1 of B and C
        # B indices: j=0,1; k=2b, 2b+1
        # V indices: j*p + k
        v_00 = 0 * p + 2 * b      # B[0, 2b]
        v_01 = 0 * p + (2 * b + 1) # B[0, 2b+1]
        v_10 = 1 * p + 2 * b      # B[1, 2b]
        v_11 = 1 * p + (2 * b + 1) # B[1, 2b+1]
        
        # C indices: i=0,1; k=2b, 2b+1
        # W indices: k*n + i
        w_00 = (2 * b) * n + 0      # C[0, 2b]
        w_01 = (2 * b + 1) * n + 0  # C[0, 2b+1]
        w_10 = (2 * b) * n + 1      # C[1, 2b]
        w_11 = (2 * b + 1) * n + 1  # C[1, 2b+1]
        
        # A indices: i=0,1; j=0,1
        # U indices: i*m + j
        u_00 = 0 * m + 0 # A[0,0]
        u_01 = 0 * m + 1 # A[0,1]
        u_10 = 1 * m + 0 # A[1,0]
        u_11 = 1 * m + 1 # A[1,1]
        
        # Strassen products:
        # M1 = A00 * (B00 + B11)
        # M2 = (A00 + A10) * B11
        # M3 = A11 * (B00 + B01)
        # M4 = (A01 - A11) * (B10 + B11)
        # M5 = (A00 + A01) * B01
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A11 - A10) * B00
        
        # M1
        U[u_00][col] = 1
        V[v_00][col] = 1
        V[v_11][col] = 1
        W[w_00][col] = 1
        W[w_11][col] = 1
        col += 1
        
        # M2
        U[u_00][col] = 1
        U[u_10][col] = 1
        V[v_11][col] = 1
        W[w_10][col] = 1
        W[w_11][col] = -1
        col += 1
        
        # M3
        U[u_11][col] = 1
        V[v_00][col] = 1
        V[v_01][col] = 1
        W[w_01][col] = 1
        W[w_11][col] = 1
        col += 1
        
        # M4
        U[u_01][col] = 1
        U[u_11][col] = -1
        V[v_10][col] = 1
        V[v_11][col] = 1
        W[w_00][col] = 1
        W[w_11][col] = -1
        col += 1
        
        # M5
        U[u_00][col] = 1
        U[u_01][col] = 1
        V[v_01][col] = 1
        W[w_00][col] = -1
        W[w_01][col] = 1
        col += 1
        
        # M6
        U[u_10][col] = 1
        U[u_00][col] = -1
        V[v_00][col] = 1
        V[v_01][col] = 1
        W[w_10][col] = 1
        W[w_11][col] = 1
        col += 1
        
        # M7
        U[u_11][col] = 1
        U[u_10][col] = -1
        V[v_00][col] = 1
        W[w_00][col] = 1
        W[w_11][col] = -1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
