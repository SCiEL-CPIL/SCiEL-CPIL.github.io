"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 39: 5 blocks of Strassen (7 mults each) + 1 block of standard (4 mults)
    rank = 5 * 7 + 4
    
    # U shape: (n*m, rank) = (4, 39)
    # V shape: (m*p, rank) = (22, 39)
    # W shape: (p*n, rank) = (22, 39)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 5 Strassen blocks for columns 0..9
    for t in range(5):
        c0 = 2 * t
        c1 = 2 * t + 1
        
        # W indices for this block
        # C[0, c0] -> idx 2*c0 + 0 = 4t
        # C[0, c1] -> idx 2*c1 + 0 = 4t+2
        # C[1, c0] -> idx 2*c0 + 1 = 4t+1
        # C[1, c1] -> idx 2*c1 + 1 = 4t+3
        
        # P1: (A00 + A11) * (B0,c0 + B1,c1)
        U[0][col] = 1
        U[3][col] = 1
        V[c0][col] = 1
        V[m*p//2 + c1][col] = 1 # V[j*p+k], j=1, k=c1 -> 11+c1. m*p//2 is 11.
        W[4*t][col] = 1
        W[4*t+3][col] = 1
        col += 1
        
        # P2: (A10 + A11) * B0,c0
        U[2][col] = 1
        U[3][col] = 1
        V[c0][col] = 1
        W[4*t+1][col] = 1 # C[1, c0]
        W[4*t+3][col] = -1 # C[1, c1]
        col += 1
        
        # P3: A00 * (B0,c1 - B1,c0)
        U[0][col] = 1
        V[c1][col] = 1
        V[m*p//2 + c0][col] = -1
        W[4*t+2][col] = 1 # C[0, c1]
        W[4*t+3][col] = -1 # C[1, c1]
        col += 1
        
        # P4: A11 * (B1,c0 - B0,c0)
        U[3][col] = 1
        V[m*p//2 + c0][col] = 1
        V[c0][col] = -1
        W[4*t][col] = -1 # C[0, c0]
        W[4*t+1][col] = 1 # C[1, c0]
        col += 1
        
        # P5: (A00 + A01) * B1,c1
        U[0][col] = 1
        U[1][col] = 1
        V[m*p//2 + c1][col] = 1
        W[4*t][col] = 1 # C[0, c0]
        W[4*t+2][col] = 1 # C[0, c1]
        col += 1
        
        # P6: (A10 - A00) * (B0,c0 + B0,c1)
        U[2][col] = 1
        U[0][col] = -1
        V[c0][col] = 1
        V[c1][col] = 1
        W[4*t+3][col] = 1 # C[1, c1]
        col += 1
        
        # P7: (A01 - A11) * (B1,c0 + B1,c1)
        U[1][col] = 1
        U[3][col] = -1
        V[m*p//2 + c0][col] = 1
        V[m*p//2 + c1][col] = 1
        W[4*t][col] = -1 # C[0, c0]
        col += 1
        
    # Last column 10
    c_last = 10
    
    # M1: A00 * B0,10
    U[0][col] = 1
    V[c_last][col] = 1
    W[2*c_last][col] = 1 # C[0, 10]
    col += 1
    
    # M2: A01 * B1,10
    U[1][col] = 1
    V[m*p//2 + c_last][col] = 1
    W[2*c_last][col] = 1 # C[0, 10]
    col += 1
    
    # M3: A10 * B0,10
    U[2][col] = 1
    V[c_last][col] = 1
    W[2*c_last + 1][col] = 1 # C[1, 10]
    col += 1
    
    # M4: A11 * B1,10
    U[3][col] = 1
    V[m*p//2 + c_last][col] = 1
    W[2*c_last + 1][col] = 1 # C[1, 10]
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
