"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Decomposition for 2x3 * 3x10 matrix multiplication.
    
    Split A (2x3) into A1 (2x2, cols 0-1) and A2 (2x1, col 2).
    Split B (3x10) into B1 (2x10, rows 0-1) and B2 (1x10, row 2).
    C = A1*B1 + A2*B2
    
    A1*B1: 2x2 * 2x10, Strassen per 2-column block: 5 mults * 5 blocks = 25
    A2*B2: 2x1 * 1x10, schoolbook: 2*10 = 20
    Total rank = 45
    """
    n, m, p = DIMENSIONS  # 2, 3, 10
    rank = 45
    U = [[0] * rank for _ in range(n * m)]  # 6 x 45
    V = [[0] * rank for _ in range(m * p)]  # 30 x 45
    W = [[0] * rank for _ in range(p * n)]  # 20 x 45
    
    # A1*B1: Strassen per 2-column block
    for j in range(5):
        base = j * 5
        col0 = 2 * j
        col1 = 2 * j + 1
        
        # M1 = (a00+a11)(b00+b11)
        U[0][base] = 1    # a00
        U[4][base] = 1    # a11
        V[col0][base] = 1          # b00
        V[10 + col1][base] = 1     # b11
        W[col0 * 2 + 0][base] = 1  # c00
        W[col1 * 2 + 1][base] = 1  # c11
        
        # M2 = (a10+a11)b00
        U[3][base + 1] = 1    # a10
        U[4][base + 1] = 1    # a11
        V[col0][base + 1] = 1 # b00
        W[col0 * 2 + 1][base + 1] = 1  # c10
        
        # M3 = a00(b10-b11)
        U[0][base + 2] = 1    # a00
        V[10 + col0][base + 2] = 1   # b10
        V[10 + col1][base + 2] = -1  # b11
        W[col1 * 2 + 0][base + 2] = 1  # c01
        
        # M4 = a11(b01+b11)
        U[4][base + 3] = 1    # a11
        V[col1][base + 3] = 1   # b01
        V[10 + col1][base + 3] = 1 # b11
        W[col0 * 2 + 0][base + 3] = 1  # c00
        W[col0 * 2 + 1][base + 3] = 1  # c10
        
        # M5 = (a00+a01)b11
        U[0][base + 4] = 1    # a00
        U[1][base + 4] = 1    # a01
        V[10 + col1][base + 4] = 1 # b11
        W[col0 * 2 + 0][base + 4] = -1  # c00
        W[col1 * 2 + 0][base + 4] = 1   # c01
    
    # A2*B2: schoolbook
    for i in range(2):
        for k in range(10):
            r = 25 + i * 10 + k
            U[i * 3 + 2][r] = 1  # a_i2
            V[20 + k][r] = 1     # b2k
            W[k * 2 + i][r] = 1  # c_ik
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
