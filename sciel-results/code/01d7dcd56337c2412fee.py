"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 55 decomposition
    # 5 blocks of M_2,2,2 (rank 7) + 1 block of M_2,1,10 (rank 20)
    rank = 5 * 7 + 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen helper for 2x2x2
    # A is 2x2, B is 2x2
    # U indices for A: i*2 + j (but in global U, it's i*3 + j)
    # V indices for B: j*10 + k (global V)
    # W indices for C: k*2 + i (global W)
    
    def add_strassen_block(base_j, base_k):
        nonlocal col
        # A cols: base_j, base_j+1
        # B rows: base_j, base_j+1
        # B cols: base_k, base_k+1
        
        # M1 = (A00+A11)(B00+B11)
        U[0*m + base_j][col] = 1
        U[1*m + base_j+1][col] = 1
        V[base_j*p + base_k][col] = 1
        V[(base_j+1)*p + (base_k+1)][col] = 1
        W[base_k*n + 0][col] = 1
        W[(base_k+1)*n + 1][col] = 1
        W[base_k*n + 1][col] = -1
        W[(base_k+1)*n + 0][col] = -1
        col += 1
        
        # M2 = (A10+A11)B00
        U[1*m + base_j][col] = 1
        U[1*m + base_j+1][col] = 1
        V[base_j*p + base_k][col] = 1
        W[base_k*n + 1][col] = 1
        W[(base_k+1)*n + 1][col] = -1
        col += 1
        
        # M3 = A00(B10-B11)
        U[0*m + base_j][col] = 1
        V[(base_j+1)*p + base_k][col] = 1
        V[(base_j+1)*p + (base_k+1)][col] = -1
        W[base_k*n + 1][col] = 1
        W[(base_k+1)*n + 1][col] = 1
        col += 1
        
        # M4 = A11(B01-B00)
        U[1*m + base_j+1][col] = 1
        V[base_j*p + (base_k+1)][col] = 1
        V[base_j*p + base_k][col] = -1
        W[base_k*n + 1][col] = 1
        W[(base_k+1)*n + 0][col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[0*m + base_j][col] = 1
        U[0*m + base_j+1][col] = 1
        V[(base_j+1)*p + (base_k+1)][col] = 1
        W[base_k*n + 0][col] = -1
        W[(base_k+1)*n + 1][col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B10)
        U[1*m + base_j][col] = 1
        U[0*m + base_j][col] = -1
        V[base_j*p + base_k][col] = 1
        V[(base_j+1)*p + base_k][col] = 1
        W[(base_k+1)*n + 1][col] = 1
        col += 1
        
        # M7 = (A01-A11)(B11+B01)
        U[0*m + base_j+1][col] = 1
        U[1*m + base_j+1][col] = -1
        V[(base_j+1)*p + (base_k+1)][col] = 1
        V[base_j*p + (base_k+1)][col] = 1
        W[base_k*n + 0][col] = 1
        W[(base_k+1)*n + 0][col] = -1
        col += 1

    # Apply 5 Strassen blocks
    for k in range(5):
        add_strassen_block(0, 2*k)
        
    # Remainder: A col 2, B row 2
    # C_ik = A_i2 * B_2k
    for i in range(2):
        for k in range(10):
            U[i*m + 2][col] = 1
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
