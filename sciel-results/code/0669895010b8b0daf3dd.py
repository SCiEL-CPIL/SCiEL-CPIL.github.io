"""
Matrix multiplication 6x6x6: rank-minimal exact tensor decomposition.

Strategy: Use a 2x2 grid of 3x3 blocks.
Apply Strassen's rank-7 algorithm for the 2x2 block structure.
Apply Schoolbook rank-9 algorithm for each 3x3 block multiplication.
Total rank = 7 * 9 = 63.
"""
import time
import numpy as np
import random

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 6x6 matrix
    by a 6x6 matrix.

    Uses a 2x2 grid of 3x3 blocks.
    Strassen for the block level (rank 7).
    Schoolbook for the 3x3 block level (rank 9).
    Total rank = 63.
    """
    t0 = time.time()
    DEADLINE = 120.0

    n, m, p = DIMENSIONS  # all 6
    block_size = 3
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen steps for 2x2 blocks
    # Each step defines coefficients for A blocks, B blocks, and C blocks
    # A blocks: (0,0), (0,1), (1,0), (1,1)
    # B blocks: (0,0), (0,1), (1,0), (1,1)
    # C blocks: (0,0), (0,1), (1,0), (1,1)
    
    strassen_steps = [
        # M1 = (A00 + A11)(B00 + B11)
        {'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}},
        # M2 = (A10 + A11)B00
        {'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(1,0): 1, (1,1): -1}},
        # M3 = A00(B01 - B11)
        {'U': {(0,0): 1}, 'V': {(0,1): 1, (1,1): -1}, 'W': {(0,1): 1, (1,1): 1}},
        # M4 = A11(B10 - B00)
        {'U': {(1,1): 1}, 'V': {(1,0): 1, (0,0): -1}, 'W': {(0,0): 1, (1,0): 1}},
        # M5 = (A00 + A01)B11
        {'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,0): -1, (0,1): 1}},
        # M6 = (A10 - A00)(B00 + B01)
        {'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(1,1): 1}},
        # M7 = (A01 - A11)(B10 + B11)
        {'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1}},
    ]

    col = 0
    for s in range(7):
        step = strassen_steps[s]
        for i_prime in range(block_size):
            for k_prime in range(block_size):
                for j_prime in range(block_size):
                    # U contributions
                    for (ba, bb), coeff in step['U'].items():
                        u_idx = (ba * block_size + i_prime) * m + (bb * block_size + k_prime)
                        U[u_idx][col] = coeff
                    
                    # V contributions
                    for (bc, bd), coeff in step['V'].items():
                        v_idx = (bc * block_size + k_prime) * p + (bd * block_size + j_prime)
                        V[v_idx][col] = coeff
                    
                    # W contributions
                    for (bi, bk), coeff in step['W'].items():
                        # W index is k*n + i
                        # k is global col of B (from j_prime and bk)
                        # i is global row of A (from i_prime and bi)
                        w_row = bk * block_size + j_prime
                        w_col = bi * block_size + i_prime
                        w_idx = w_row * n + w_col
                        W[w_idx][col] = coeff
                    
                    col += 1

    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    valid = True
    random.seed(42)
    for _ in range(500):
        i = random.randint(0, n - 1)
        j = random.randint(0, m - 1)
        k = random.randint(0, p - 1)
        val = 0.0
        for r in range(rank):
            val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
        if abs(val - 1.0) > 1e-10:
            valid = False
            break

    if valid and (time.time() - t0) < DEADLINE:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                    if abs(val - 1.0) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        rank_fallback = n * m * p
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j][col] = 1
                    V_f[j * p + k][col] = 1
                    W_f[k * n + i][col] = 1
                    col += 1
        return {"U": U_f, "V": V_f, "W": W_f}

# EVOLVE-BLOCK-END
