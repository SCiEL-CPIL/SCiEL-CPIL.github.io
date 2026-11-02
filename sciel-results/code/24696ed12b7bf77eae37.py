"""
Rank-minimal tensor decomposition for 2x2x9 matrix multiplication.
Uses 4 Strassen blocks (rank 7 each) for column pairs (0,1), (2,3), (4,5), (6,7)
plus 1 schoolbook block (rank 4) for column 8.
Total rank = 4*7 + 4 = 32.
"""
import time

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    
    rank = 32
    U = [[0.0] * rank for _ in range(n * m)]   # 4 x 32
    V = [[0.0] * rank for _ in range(m * p)]   # 18 x 32
    W = [[0.0] * rank for _ in range(p * n)]   # 18 x 32
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        # M1: (a00+a11)(b0k0+b1k1) -> C0k0, C1k1
        U[0][col] = 1.0; U[3][col] = 1.0
        V[0*p + k0][col] = 1.0; V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = 1.0; W[k1*n + 1][col] = 1.0
        col += 1
        # M2: (a10+a11)*b0k0 -> C1k0, -C1k1
        U[2][col] = 1.0; U[3][col] = 1.0
        V[0*p + k0][col] = 1.0
        W[k0*n + 1][col] = 1.0; W[k1*n + 1][col] = -1.0
        col += 1
        # M3: a00*(b0k1-b1k1) -> C0k1, C1k1
        U[0][col] = 1.0
        V[0*p + k1][col] = 1.0; V[1*p + k1][col] = -1.0
        W[k1*n + 0][col] = 1.0; W[k1*n + 1][col] = 1.0
        col += 1
        # M4: a11*(b1k0-b0k0) -> C0k0, C1k0
        U[3][col] = 1.0
        V[1*p + k0][col] = 1.0; V[0*p + k0][col] = -1.0
        W[k0*n + 0][col] = 1.0; W[k0*n + 1][col] = 1.0
        col += 1
        # M5: (a00+a01)*b1k1 -> -C0k0, C0k1
        U[0][col] = 1.0; U[1][col] = 1.0
        V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = -1.0; W[k1*n + 0][col] = 1.0
        col += 1
        # M6: (a10-a00)*(b0k0+b0k1) -> C1k1
        U[2][col] = 1.0; U[0][col] = -1.0
        V[0*p + k0][col] = 1.0; V[0*p + k1][col] = 1.0
        W[k1*n + 1][col] = 1.0
        col += 1
        # M7: (a01-a11)*(b1k0+b1k1) -> C0k0
        U[1][col] = 1.0; U[3][col] = -1.0
        V[1*p + k0][col] = 1.0; V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = 1.0
        col += 1

    def add_schoolbook(k):
        nonlocal col
        U[0][col] = 1.0
        V[0*p + k][col] = 1.0
        W[k*n + 0][col] = 1.0
        col += 1
        U[1][col] = 1.0
        V[1*p + k][col] = 1.0
        W[k*n + 0][col] = 1.0
        col += 1
        U[2][col] = 1.0
        V[0*p + k][col] = 1.0
        W[k*n + 1][col] = 1.0
        col += 1
        U[3][col] = 1.0
        V[1*p + k][col] = 1.0
        W[k*n + 1][col] = 1.0
        col += 1

    # 4 Strassen blocks for column pairs
    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    # 1 schoolbook block for column 8
    add_schoolbook(8)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
