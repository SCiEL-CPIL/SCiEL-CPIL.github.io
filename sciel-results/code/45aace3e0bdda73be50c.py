"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        # k0, k1 are the column indices in B
        # U indices: i*m + j. i in 0..1, j in 0..1
        # V indices: j*p + k. j in 0..1, k in k0, k1
        # W indices: k*n + i. k in k0, k1, i in 0..1
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0][col] += 1; U[3][col] += 1
        V[0*p + k0][col] += 1; V[1*p + k1][col] += 1
        W[k0*n + 0][col] += 1; W[k1*n + 1][col] += 1
        col += 1
        
        # M2 = (A10 + A11) * B0k0
        U[2][col] += 1; U[3][col] += 1
        V[0*p + k0][col] += 1
        W[k0*n + 1][col] += 1
        col += 1
        
        # M3 = A00 * (B0k1 - B1k1)
        U[0][col] += 1
        V[0*p + k1][col] += 1; V[1*p + k1][col] -= 1
        W[k1*n + 0][col] += 1
        col += 1
        
        # M4 = A11 * (B1k0 - B0k0)
        U[3][col] += 1
        V[1*p + k0][col] += 1; V[0*p + k0][col] -= 1
        W[k0*n + 1][col] += 1; W[k1*n + 1][col] -= 1
        col += 1
        
        # M5 = (A00 + A01) * B1k1
        U[0][col] += 1; U[1][col] += 1
        V[1*p + k1][col] += 1
        W[k1*n + 0][col] += 1
        col += 1
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[2][col] += 1; U[0][col] -= 1
        V[0*p + k0][col] += 1; V[0*p + k1][col] += 1
        W[k0*n + 1][col] -= 1; W[k1*n + 1][col] -= 1
        col += 1
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[1][col] += 1; U[3][col] -= 1
        V[1*p + k0][col] += 1; V[1*p + k1][col] += 1
        W[k0*n + 0][col] += 1
        col += 1

    def add_basic(k):
        nonlocal col
        # A00 * B0k * C0k
        U[0][col] += 1
        V[0*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        # A01 * B1k * C0k
        U[1][col] += 1
        V[1*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        # A10 * B0k * C1k
        U[2][col] += 1
        V[0*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
        
        # A11 * B1k * C1k
        U[3][col] += 1
        V[1*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1

    # Block 1: Cols 0, 1
    add_strassen(0, 1)
    
    # Block 2: Cols 2, 3
    add_strassen(2, 3)
    
    # Block 3: Col 4
    add_basic(4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
