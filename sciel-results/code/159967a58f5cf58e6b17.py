"""Matrix multiplication 2x4 x 4x6 — rank-minimal exact tensor decomposition."""
import time
import sys
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _verify(n, m, p, U, V, W):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for c in range(rank):
        uc = np.array([U[i][c] for i in range(n * m)], dtype=float)
        vc = np.array([V[i][c] for i in range(m * p)], dtype=float)
        wc = np.array([W[i][c] for i in range(p * n)], dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] += uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
    target = np.zeros_like(T)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T, target, atol=1e-9)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_block_decomposition(n, m, p):
    """
    Decompose M_{n,m,p} using block Strassen.
    For (2,4,6): 
    - Split A (2x4) into 2 blocks of 2x2 (columns 0-1, 2-3).
    - Split B (4x6) into 3 blocks of 2x2 (columns 0-1, 2-3, 4-5).
    - Each block multiplication is 2x2 * 2x2 -> 2x2, solved by Strassen (rank 7).
    - Total rank = 2 * 3 * 7 = 42.
    """
    # Dimensions
    n = 2
    m = 4
    p = 6
    
    # Block parameters
    a_blocks = 2  # Number of A column blocks (m / 2)
    b_col_blocks = 3  # Number of B column blocks (p / 2)
    block_size = 2
    
    rank = a_blocks * b_col_blocks * 7
    if rank >= n * m * p:
        return None  # No improvement, fallback to schoolbook

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 * 2x2 -> 2x2
    # Let M = [[m00, m01], [m10, m11]], N = [[n00, n01], [n10, n11]]
    # Products:
    # P1 = (m00 + m11) * (n00 + n11)
    # P2 = (m10 + m11) * n00
    # P3 = m00 * (n01 - n11)
    # P4 = m11 * (n10 - n00)
    # P5 = (m00 + m01) * n11
    # P6 = (m10 - m00) * (n00 + n01)
    # P7 = (m01 - m11) * (n10 + n11)
    
    # Output R = [[r00, r01], [r10, r11]]
    # r00 = P1 + P4 - P5 + P7
    # r01 = P3 + P5
    # r10 = P2 + P4
    # r11 = P1 - P2 + P3 + P6
    
    # U coefficients: contribution to M entries (i_loc, j_loc) for each P_s
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},  # P1
        {(1, 0): 1, (1, 1): 1},  # P2
        {(0, 0): 1},              # P3
        {(1, 1): 1},              # P4
        {(0, 0): 1, (0, 1): 1},  # P5
        {(1, 0): 1, (0, 0): -1}, # P6
        {(0, 1): 1, (1, 1): -1}  # P7
    ]
    
    # V coefficients: contribution to N entries (j_loc, k_loc) for each P_s
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},  # P1
        {(0, 0): 1},              # P2
        {(0, 1): 1, (1, 1): -1}, # P3
        {(1, 0): 1, (0, 0): -1}, # P4
        {(1, 1): 1},              # P5
        {(0, 0): 1, (0, 1): 1},  # P6
        {(1, 0): 1, (1, 1): 1}   # P7
    ]
    
    # W coefficients: contribution to R entries (i_loc, k_loc) for each P_s
    W_coeffs = [
        {(0, 0): 1, (1, 1): 1},  # P1 contributes to r00 (+1), r11 (+1)
        {(1, 0): 1, (1, 1): -1}, # P2 contributes to r10 (+1), r11 (-1)
        {(0, 1): 1, (1, 1): 1},  # P3 contributes to r01 (+1), r11 (+1)
        {(0, 0): 1, (1, 0): 1},  # P4 contributes to r00 (+1), r10 (+1)
        {(0, 0): -1, (0, 1): 1}, # P5 contributes to r00 (-1), r01 (+1)
        {(1, 1): 1},             # P6 contributes to r11 (+1)
        {(0, 0): 1}              # P7 contributes to r00 (+1)
    ]

    col = 0
    for a_blk in range(a_blocks):
        for b_blk in range(b_col_blocks):
            for s in range(7):
                # Update U
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a_blk * block_size + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # Update V
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a_blk * block_size + j_loc
                    k = b_blk * block_size + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # Update W
                for (i_loc, k_loc), coeff in W_coeffs[s].items():
                    i = i_loc
                    k = b_blk * block_size + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1

    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 110  # safety margin before 150s limit
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 48
    
    # Try the block Strassen decomposition
    candidate = _strassen_block_decomposition(n, m, p)
    if candidate is not None:
        cand_rank = len(candidate['U'][0])
        if cand_rank < best_rank:
            # Verify correctness
            if _verify(n, m, p, candidate['U'], candidate['V'], candidate['W']):
                best = candidate
                best_rank = cand_rank
            else:
                # If verification fails, stick to schoolbook
                pass
    
    return best

# EVOLVE-BLOCK-END
