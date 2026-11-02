"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using 7 blocks of 2x2 Strassen
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen coefficients for 2x2 by 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B11-B00)
    # M4 = A11(B00-B10)
    # M5 = (A00+A10)B11
    # M6 = (A11-A10)(B00+B10)
    # M7 = (A01-A00)(B11+B01)
    
    # U indices: 0:A00, 1:A01, 2:A10, 3:A11
    # V indices for block b: 
    #   B00: 2b, B01: 2b+1, B10: 14+2b, B11: 15+2b
    # W indices for block b:
    #   C00: 4b, C01: 4b+1, C10: 4b+2, C11: 4b+3
    
    for b in range(7):
        # V indices
        v00 = 2 * b
        v01 = 2 * b + 1
        v10 = 14 + 2 * b
        v11 = 15 + 2 * b
        
        # W indices
        w00 = 4 * b
        w01 = 4 * b + 1
        w10 = 4 * b + 2
        w11 = 4 * b + 3
        
        # M1
        U[0][col] += 1; U[3][col] += 1
        V[v00][col] += 1; V[v11][col] += 1
        W[w00][col] += 1; W[w10][col] -= 1; W[w11][col] += 1
        col += 1
        
        # M2
        U[2][col] += 1; U[3][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1; W[w11][col] += 1
        col += 1
        
        # M3
        U[0][col] += 1
        V[v11][col] += 1; V[v00][col] -= 1
        W[w10][col] -= 1; W[w01][col] += 1
        col += 1
        
        # M4
        U[3][col] += 1
        V[v00][col] += 1; V[v10][col] -= 1
        W[w00][col] += 1; W[w11][col] -= 1
        col += 1
        
        # M5
        U[0][col] += 1; U[2][col] += 1
        V[v11][col] += 1
        W[w00][col] -= 1; W[w10][col] += 1
        col += 1
        
        # M6
        U[3][col] += 1; U[2][col] -= 1
        V[v00][col] += 1; V[v10][col] += 1
        W[w01][col] += 1; W[w11][col] -= 1
        col += 1
        
        # M7
        U[1][col] += 1; U[0][col] -= 1
        V[v11][col] += 1; V[v01][col] += 1
        W[w00][col] += 1; W[w01][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
