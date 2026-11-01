"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 5]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin for the 150s limit

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=2, m=2, p=5):
    """
    Verify if the given U, V, W correctly decompose the matrix multiplication tensor.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank) with convention W[k*n+i, r]
    
    The tensor entry T[i, j, k] is 1 if (i,j,k) is a valid index.
    The decomposition says:
    T[i, j, k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                computed = 0.0
                for r in range(rank):
                    computed += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(computed - target) > 1e-9:
                    return False
    return True

def _construct_rank18(n=2, m=2, p=5):
    """
    Construct a rank-18 decomposition using Strassen for 2x2x2 (cols 0-1),
    Strassen for 2x2x2 (cols 2-3), and Schoolbook for 2x2x1 (col 4).
    """
    rank = 18
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Helper to add a rank-1 term
    # u_idx, v_idx, w_idx, col
    def add_term(u_idx, v_idx, w_idx, col, sign=1):
        U[u_idx][col] += sign
        V[v_idx][col] += sign
        W[w_idx][col] += sign

    # --- Part 1: Strassen for cols 0, 1 ---
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=2, C11=3
    
    # P1 = A00 * (B00 + B11)
    # U: A00 (0), V: B00(0), B11(3), W: C00(0), C11(3)
    # We need to represent this as sum of rank 1 terms? No, Strassen terms are products of linear forms.
    # In tensor decomposition, each "multiplication" is a rank 1 tensor.
    # P1 corresponds to one column in U, V, W.
    # U[0] (A00) is 1.
    # V[0] (B00) is 1, V[3] (B11) is 1.
    # W[0] (C00) is 1, W[3] (C11) is 1.
    
    # Col 0: P1
    U[0][0] = 1
    V[0][0] = 1
    V[3][0] = 1
    W[0][0] = 1
    W[3][0] = 1
    
    # P2 = (A10 + A11) * B00
    # U: A10(2), A11(3), V: B00(0), W: C10(2)
    # Col 1: P2
    U[2][1] = 1
    U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # P3 = (A00 + A01) * B11
    # U: A00(0), A01(1), V: B11(3), W: C01(1), C11(3)
    # Col 2: P3
    U[0][2] = 1
    U[1][2] = 1
    V[3][2] = 1
    W[1][2] = 1
    W[3][2] = 1
    
    # P4 = A11 * (B01 - B00)
    # U: A11(3), V: B01(1), B00(0), W: C00(0)
    # Col 3: P4
    U[3][3] = 1
    V[1][3] = 1
    V[0][3] = -1
    W[0][3] = 1
    
    # P5 = A00 * (B10 - B11)
    # U: A00(0), V: B10(2), B11(3), W: C00(0), C10(2)
    # Col 4: P5
    U[0][4] = 1
    V[2][4] = 1
    V[3][4] = -1
    W[0][4] = -1
    W[2][4] = 1
    
    # P6 = (A10 - A00) * B00
    # U: A10(2), A00(0), V: B00(0), W: C11(3)
    # Col 5: P6
    U[2][5] = 1
    U[0][5] = -1
    V[0][5] = 1
    W[3][5] = 1
    
    # P7 = (A01 - A11) * B11
    # U: A01(1), A11(3), V: B11(3), W: C00(0), C01(1)
    # Col 6: P7
    U[1][6] = 1
    U[3][6] = -1
    V[3][6] = 1
    W[0][6] = -1
    W[1][6] = 1

    # --- Part 2: Strassen for cols 2, 3 ---
    # B indices: B02=4, B03=5, B12=7, B13=8
    # C indices: C02=4, C03=5, C12=6, C13=7
    
    # Q1 = A00 * (B02 + B13)
    # Col 7: Q1
    U[0][7] = 1
    V[4][7] = 1
    V[8][7] = 1
    W[4][7] = 1
    W[7][7] = 1
    
    # Q2 = (A10 + A11) * B02
    # Col 8: Q2
    U[2][8] = 1
    U[3][8] = 1
    V[4][8] = 1
    W[6][8] = 1
    
    # Q3 = (A00 + A01) * B13
    # Col 9: Q3
    U[0][9] = 1
    U[1][9] = 1
    V[8][9] = 1
    W[5][9] = 1
    W[7][9] = 1
    
    # Q4 = A11 * (B03 - B02)
    # Col 10: Q4
    U[3][10] = 1
    V[5][10] = 1
    V[4][10] = -1
    W[4][10] = 1
    
    # Q5 = A00 * (B12 - B13)
    # Col 11: Q5
    U[0][11] = 1
    V[7][11] = 1
    V[8][11] = -1
    W[4][11] = -1
    W[6][11] = 1
    
    # Q6 = (A10 - A00) * B02
    # Col 12: Q6
    U[2][12] = 1
    U[0][12] = -1
    V[4][12] = 1
    W[7][12] = 1
    
    # Q7 = (A01 - A11) * B13
    # Col 13: Q7
    U[1][13] = 1
    U[3][13] = -1
    V[8][13] = 1
    W[4][13] = -1
    W[5][13] = 1

    # --- Part 3: Schoolbook for col 4 ---
    # B indices: B04=6, B14=9
    # C indices: C04=8, C14=9
    
    # R1 = A00 * B04
    # Col 14: R1
    U[0][14] = 1
    V[6][14] = 1
    W[8][14] = 1
    
    # R2 = A01 * B14
    # Col 15: R2
    U[1][15] = 1
    V[9][15] = 1
    W[8][15] = 1
    
    # R3 = A10 * B04
    # Col 16: R3
    U[2][16] = 1
    V[6][16] = 1
    W[9][16] = 1
    
    # R4 = A11 * B14
    # Col 17: R4
    U[3][17] = 1
    V[9][17] = 1
    W[9][17] = 1

    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    return None

def _construct_schoolbook(n=2, m=2, p=5):
    """
    Construct the standard schoolbook decomposition of rank n*m*p.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    
    # Try rank 18 first
    result = _construct_rank18(n, m, p)
    if result is not None:
        return result
    
    # Fallback: Schoolbook (rank 20)
    return _construct_schoolbook(n, m, p)

# EVOLVE-BLOCK-END
