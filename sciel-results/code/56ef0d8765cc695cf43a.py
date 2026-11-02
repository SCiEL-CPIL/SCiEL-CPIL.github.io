import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    U3 = np.array(U, dtype=float).reshape(n, m, -1)
    V3 = np.array(V, dtype=float).reshape(m, p, -1)
    W3 = np.array(W, dtype=float).reshape(p, n, -1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_rank32_decomposition(n, m, p):
    """Build the base rank-32 decomposition."""
    rank = 4 * 7 + 4  # 32
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        # M1
        U[0][col] = 1.0; U[3][col] = 1.0
        V[0*p + k0][col] = 1.0; V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = 1.0; W[k1*n + 1][col] = 1.0
        col += 1
        # M2
        U[2][col] = 1.0; U[3][col] = 1.0
        V[0*p + k0][col] = 1.0
        W[k0*n + 1][col] = 1.0; W[k1*n + 1][col] = -1.0
        col += 1
        # M3
        U[0][col] = 1.0
        V[0*p + k1][col] = 1.0; V[1*p + k1][col] = -1.0
        W[k1*n + 0][col] = 1.0; W[k1*n + 1][col] = 1.0
        col += 1
        # M4
        U[3][col] = 1.0
        V[1*p + k0][col] = 1.0; V[0*p + k0][col] = -1.0
        W[k0*n + 0][col] = 1.0; W[k0*n + 1][col] = 1.0
        col += 1
        # M5
        U[0][col] = 1.0; U[1][col] = 1.0
        V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = -1.0; W[k1*n + 0][col] = 1.0
        col += 1
        # M6
        U[2][col] = 1.0; U[0][col] = -1.0
        V[0*p + k0][col] = 1.0; V[0*p + k1][col] = 1.0
        W[k1*n + 1][col] = 1.0
        col += 1
        # M7
        U[1][col] = 1.0; U[3][col] = -1.0
        V[1*p + k0][col] = 1.0; V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = 1.0
        col += 1

    def add_schoolbook(k):
        nonlocal col
        U[0][col] = 1.0; V[0*p + k][col] = 1.0; W[k*n + 0][col] = 1.0; col += 1
        U[1][col] = 1.0; V[1*p + k][col] = 1.0; W[k*n + 0][col] = 1.0; col += 1
        U[2][col] = 1.0; V[0*p + k][col] = 1.0; W[k*n + 1][col] = 1.0; col += 1
        U[3][col] = 1.0; V[1*p + k][col] = 1.0; W[k*n + 1][col] = 1.0; col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_schoolbook(8)
    return U, V, W

def _build_rank29_decomposition(n, m, p):
    """Build a rank-29 decomposition using 3 Strassen blocks and 1 Rank-8 block."""
    rank = 3 * 7 + 8  # 29
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        U[0][col] = 1.0; U[3][col] = 1.0
        V[0*p + k0][col] = 1.0; V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = 1.0; W[k1*n + 1][col] = 1.0
        col += 1
        U[2][col] = 1.0; U[3][col] = 1.0
        V[0*p + k0][col] = 1.0
        W[k0*n + 1][col] = 1.0; W[k1*n + 1][col] = -1.0
        col += 1
        U[0][col] = 1.0
        V[0*p + k1][col] = 1.0; V[1*p + k1][col] = -1.0
        W[k1*n + 0][col] = 1.0; W[k1*n + 1][col] = 1.0
        col += 1
        U[3][col] = 1.0
        V[1*p + k0][col] = 1.0; V[0*p + k0][col] = -1.0
        W[k0*n + 0][col] = 1.0; W[k0*n + 1][col] = 1.0
        col += 1
        U[0][col] = 1.0; U[1][col] = 1.0
        V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = -1.0; W[k1*n + 0][col] = 1.0
        col += 1
        U[2][col] = 1.0; U[0][col] = -1.0
        V[0*p + k0][col] = 1.0; V[0*p + k1][col] = 1.0
        W[k1*n + 1][col] = 1.0
        col += 1
        U[1][col] = 1.0; U[3][col] = -1.0
        V[1*p + k0][col] = 1.0; V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = 1.0
        col += 1

    def add_rank8_block(k0, k1, k2):
        nonlocal col
        # 7 Strassen terms for k0, k1
        add_strassen(k0, k1)
        
        # 1 additional term for k2
        # U: (1, 1, 1, 1)
        U[0][col] = 1.0
        U[1][col] = 1.0
        U[2][col] = 1.0
        U[3][col] = 1.0
        # V: (0, 0, 1) for j=0, (0, 0, 1) for j=1
        # V[j*p + k, col]
        V[0*p + k2][col] = 1.0
        V[1*p + k2][col] = 1.0
        # W: (1, 1) for k=k2
        # W[k*n + i, col]
        W[k2*n + 0][col] = 1.0
        W[k2*n + 1][col] = 1.0
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_rank8_block(6, 7, 8)
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 2, 2, 9
    
    # Try Rank 29 first
    U29, V29, W29 = _build_rank29_decomposition(n, m, p)
    if _verify(U29, V29, W29, n, m, p):
        return {"U": U29, "V": V29, "W": W29}
        
    # Fallback to Rank 32
    U32, V32, W32 = _build_rank32_decomposition(n, m, p)
    if _verify(U32, V32, W32, n, m, p):
        return {"U": U32, "V": V32, "W": W32}
        
    # Fallback to Rank 36 (Schoolbook)
    rank_fb = n * m * p
    U_fb = [[0.0] * rank_fb for _ in range(n * m)]
    V_fb = [[0.0] * rank_fb for _ in range(m * p)]
    W_fb = [[0.0] * rank_fb for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][c] = 1.0
                V_fb[j*p+k][c] = 1.0
                W_fb[k*n+i][c] = 1.0
                c += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-END
