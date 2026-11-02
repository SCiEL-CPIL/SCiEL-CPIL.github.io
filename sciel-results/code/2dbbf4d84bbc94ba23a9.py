"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition:
    # 3 blocks of 2x2x2 (Strassen, rank 7)
    # 1 block of 2x2x1 (Standard, rank 4)
    # Total rank = 3 * 7 + 4 = 25
    
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block for B columns k0, k0+1
    def add_strassen_block(k0):
        nonlocal col
        # U indices: 0:a11, 1:a12, 2:a21, 3:a22
        # V indices: 
        #   b11: 0*p + k0
        #   b12: 0*p + k0+1
        #   b21: 1*p + k0
        #   b22: 1*p + k0+1
        # W indices:
        #   C11: k0*n + 0
        #   C12: (k0+1)*n + 0
        #   C21: k0*n + 1
        #   C22: (k0+1)*n + 1
        
        v_b11 = k0
        v_b12 = k0 + 1
        v_b21 = p + k0
        v_b22 = p + k0 + 1
        
        w_c11 = k0 * n
        w_c12 = (k0 + 1) * n
        w_c21 = k0 * n + 1
        w_c22 = (k0 + 1) * n + 1
        
        # M1 = a11 * (b11 + b22)
        U[0][col] = 1
        V[v_b11][col] = 1
        V[v_b22][col] = 1
        W[w_c11][col] = 1
        W[w_c22][col] = 1
        col += 1
        
        # M2 = (a11 + a12) * b22
        U[0][col] = 1
        U[1][col] = 1
        V[v_b22][col] = 1
        W[w_c12][col] = 1
        W[w_c22][col] = -1
        col += 1
        
        # M3 = (a21 + a22) * (b11 + b12)
        U[2][col] = 1
        U[3][col] = 1
        V[v_b11][col] = 1
        V[v_b12][col] = 1
        W[w_c21][col] = 1
        W[w_c22][col] = 1
        col += 1
        
        # M4 = a22 * (b22 - b12)
        U[3][col] = 1
        V[v_b22][col] = 1
        V[v_b12][col] = -1
        W[w_c11][col] = 1
        W[w_c22][col] = 1
        col += 1
        
        # M5 = (a11 + a21) * b11
        U[0][col] = 1
        U[2][col] = 1
        V[v_b11][col] = 1
        W[w_c21][col] = 1
        W[w_c22][col] = 1
        col += 1
        
        # M6 = (a12 - a21) * (b21 + b22)
        U[1][col] = 1
        U[2][col] = -1
        V[v_b21][col] = 1
        V[v_b22][col] = 1
        W[w_c11][col] = -1
        W[w_c12][col] = 1
        col += 1
        
        # M7 = (a11 - a12) * (b11 + b12)
        U[0][col] = 1
        U[1][col] = -1
        V[v_b11][col] = 1
        V[v_b12][col] = 1
        W[w_c11][col] = 1
        W[w_c12][col] = 1
        col += 1

    # Helper to add a 2x2x1 standard block for B column k0
    def add_standard_block(k0):
        nonlocal col
        v_b11 = k0
        v_b21 = p + k0
        
        w_c11 = k0 * n
        w_c21 = k0 * n + 1
        
        # M1 = a11 * b11
        U[0][col] = 1
        V[v_b11][col] = 1
        W[w_c11][col] = 1
        col += 1
        
        # M2 = a12 * b21
        U[1][col] = 1
        V[v_b21][col] = 1
        W[w_c11][col] = 1
        col += 1
        
        # M3 = a21 * b11
        U[2][col] = 1
        V[v_b11][col] = 1
        W[w_c21][col] = 1
        col += 1
        
        # M4 = a22 * b21
        U[3][col] = 1
        V[v_b21][col] = 1
        W[w_c21][col] = 1
        col += 1

    # Apply blocks
    # Columns 0, 1
    add_strassen_block(0)
    # Columns 2, 3
    add_strassen_block(2)
    # Columns 4, 5
    add_strassen_block(4)
    # Column 6
    add_standard_block(6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
