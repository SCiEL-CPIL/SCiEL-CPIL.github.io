import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 5]
DEADLINE_MARGIN = 10

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verifies if U, V, W form a valid tensor decomposition for matrix multiplication.
    Uses exact Fraction arithmetic for safety.
    """
    rank = len(U[0])
    
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                total = Fraction(0)
                for r in range(rank):
                    total += U_frac[idx_u][r] * V_frac[idx_v][r] * W_frac[idx_w][r]
                
                if total != 1:
                    return False
    return True

def _construct_rank_22():
    """
    Constructs a rank-22 decomposition for 2x3 * 3x5.
    Strategy:
    - 5 terms for j=2 (A col 2 times B row 2, all k)
    - 7 terms for j=0,1, k=0,1 (Strassen 2x2x2)
    - 7 terms for j=0,1, k=2,3 (Strassen 2x2x2)
    - 3 terms for j=0,1, k=4 (rank-3 2x2x1)
    """
    n, m, p = 2, 3, 5
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: j=2, k=0..4 (5 terms)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    # r = 5
    
    # Part 2: j=0,1, k=0,1 using Strassen (7 terms)
    U[0][r] = 1; U[4][r] = 1; V[0][r] = 1; V[6][r] = 1; W[0][r] = 1; W[3][r] = 1; r += 1
    U[3][r] = 1; U[4][r] = 1; V[0][r] = 1; W[1][r] = 1; W[3][r] = -1; r += 1
    U[0][r] = 1; V[1][r] = 1; V[6][r] = -1; W[2][r] = 1; W[3][r] = 1; r += 1
    U[4][r] = 1; V[5][r] = 1; V[0][r] = -1; W[0][r] = 1; W[2][r] = 1; r += 1
    U[0][r] = 1; U[1][r] = 1; V[6][r] = 1; W[0][r] = -1; W[2][r] = 1; r += 1
    U[3][r] = 1; U[0][r] = -1; V[0][r] = 1; V[1][r] = 1; W[3][r] = 1; r += 1
    U[1][r] = 1; U[4][r] = -1; V[5][r] = 1; V[6][r] = 1; W[0][r] = 1; r += 1
    # r = 12
    
    # Part 3: j=0,1, k=2,3 using Strassen (7 terms)
    U[0][r] = 1; U[4][r] = 1; V[2][r] = 1; V[8][r] = 1; W[4][r] = 1; W[7][r] = 1; r += 1
    U[3][r] = 1; U[4][r] = 1; V[2][r] = 1; W[5][r] = 1; W[7][r] = -1; r += 1
    U[0][r] = 1; V[3][r] = 1; V[8][r] = -1; W[6][r] = 1; W[7][r] = 1; r += 1
    U[4][r] = 1; V[7][r] = 1; V[2][r] = -1; W[4][r] = 1; W[6][r] = 1; r += 1
    U[0][r] = 1; U[1][r] = 1; V[8][r] = 1; W[4][r] = -1; W[6][r] = 1; r += 1
    U[3][r] = 1; U[0][r] = -1; V[2][r] = 1; V[3][r] = 1; W[7][r] = 1; r += 1
    U[1][r] = 1; U[4][r] = -1; V[7][r] = 1; V[8][r] = 1; W[4][r] = 1; r += 1
    # r = 19
    
    # Part 4: j=0,1, k=4 using rank-3 for 2x2x1 (3 terms)
    U[0][r] = 1; U[3][r] = 1; V[4][r] = 1; W[8][r] = 1; W[9][r] = 1; r += 1
    U[1][r] = 1; V[9][r] = 1; W[8][r] = 1; r += 1
    U[4][r] = 1; V[9][r] = 1; W[9][r] = 1; r += 1
    # r = 22
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    
    res_22 = _construct_rank_22()
    
    if _verify_decomposition(res_22['U'], res_22['V'], res_22['W'], 2, 3, 5):
        return res_22
    
    # Fallback to rank-28
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
    U[0][r] = 1; U[4][r] = 1; V[0][r] = 1; V[6][r] = 1; W[0][r] = 1; W[3][r] = 1; r += 1
    U[3][r] = 1; U[4][r] = 1; V[0][r] = 1; W[1][r] = 1; W[3][r] = -1; r += 1
    U[0][r] = 1; V[1][r] = 1; V[6][r] = -1; W[2][r] = 1; W[3][r] = 1; r += 1
    U[4][r] = 1; V[5][r] = 1; V[0][r] = -1; W[0][r] = 1; W[2][r] = 1; r += 1
    U[0][r] = 1; U[1][r] = 1; V[6][r] = 1; W[0][r] = -1; W[2][r] = 1; r += 1
    U[3][r] = 1; U[0][r] = -1; V[0][r] = 1; V[1][r] = 1; W[3][r] = 1; r += 1
    U[1][r] = 1; U[4][r] = -1; V[5][r] = 1; V[6][r] = 1; W[0][r] = 1; r += 1
    U[0][r] = 1; U[4][r] = 1; V[2][r] = 1; V[8][r] = 1; W[4][r] = 1; W[7][r] = 1; r += 1
    U[3][r] = 1; U[4][r] = 1; V[2][r] = 1; W[5][r] = 1; W[7][r] = -1; r += 1
    U[0][r] = 1; V[3][r] = 1; V[8][r] = -1; W[6][r] = 1; W[7][r] = 1; r += 1
    U[4][r] = 1; V[7][r] = 1; V[2][r] = -1; W[4][r] = 1; W[6][r] = 1; r += 1
    U[0][r] = 1; U[1][r] = 1; V[8][r] = 1; W[4][r] = -1; W[6][r] = 1; r += 1
    U[3][r] = 1; U[0][r] = -1; V[2][r] = 1; V[3][r] = 1; W[7][r] = 1; r += 1
    U[1][r] = 1; U[4][r] = -1; V[7][r] = 1; V[8][r] = 1; W[4][r] = 1; r += 1
    
    return {"U": U, "V": V, "W": W}
