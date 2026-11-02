"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 * (p/2) because we have p/2 blocks of 2x2 multiplications
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Indices in A (U): a=0, b=1, c=2, d=3
    # Indices in B block (V): e=2j, f=2j+1, g=16+2j, h=16+2j+1 (for block j)
    # Indices in C block (W): c00=4j, c10=4j+1, c01=4j+2, c11=4j+3 (for block j)
    
    for j in range(num_blocks):
        col_base = j * 7
        v_e = 2 * j
        v_f = 2 * j + 1
        v_g = 16 + 2 * j
        v_h = 16 + 2 * j + 1
        
        w_c00 = 4 * j
        w_c10 = 4 * j + 1
        w_c01 = 4 * j + 2
        w_c11 = 4 * j + 3
        
        # M1
        c = col_base
        U[0][c] = 1; U[3][c] = 1
        V[v_e][c] = 1; V[v_h][c] = 1
        W[w_c00][c] = 1; W[w_c11][c] = 1
        
        # M2
        c = col_base + 1
        U[2][c] = 1; U[3][c] = 1
        V[v_e][c] = 1
        W[w_c10][c] = 1; W[w_c11][c] = -1
        
        # M3
        c = col_base + 2
        U[0][c] = 1
        V[v_f][c] = 1; V[v_h][c] = -1
        W[w_c01][c] = 1; W[w_c11][c] = 1
        
        # M4
        c = col_base + 3
        U[3][c] = 1
        V[v_g][c] = 1; V[v_e][c] = -1
        W[w_c00][c] = 1; W[w_c10][c] = 1
        
        # M5
        c = col_base + 4
        U[0][c] = 1; U[1][c] = 1
        V[v_h][c] = 1
        W[w_c00][c] = -1; W[w_c01][c] = 1
        
        # M6
        c = col_base + 5
        U[2][c] = 1; U[0][c] = -1
        V[v_e][c] = 1; V[v_f][c] = 1
        W[w_c11][c] = 1
        
        # M7
        c = col_base + 6
        U[1][c] = 1; U[3][c] = -1
        V[v_g][c] = 1; V[v_h][c] = 1
        W[w_c00][c] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
