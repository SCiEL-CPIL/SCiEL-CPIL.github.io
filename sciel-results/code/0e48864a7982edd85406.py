"""Matrix multiplication tensor decomposition for 2x3x8."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin

    # Rank 44 construction:
    # Split B (3x8) into B_top (2x8) and B_bottom (1x8).
    # Split A (2x3) into A_left (2x2) and A_right (2x1).
    # C = A_left * B_top + A_right * B_bottom.
    
    # Part 1: A_right (2x1) * B_bottom (1x8)
    # This is 2x1x8. Rank 16.
    # u = i*3 + 2 (i in 0..1)
    # v = 2*8 + k (k in 0..7) -> 16..23
    # w = i*8 + k (i in 0..1, k in 0..7)
    
    rank = 44
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    # Part 1: 16 slots
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            
    # Part 2: A_left (2x2) * B_top (2x8)
    # Split B_top into 4 blocks of 2x2 based on k.
    # k = 0..7. Blocks: (0,1), (2,3), (4,5), (6,7).
    # For each block q (0..3), use Strassen (rank 7).
    
    # Strassen coefficients for 2x2 * 2x2
    # m1 = (a00 + a11)(b00 + b11)
    # m2 = (a10 + a11)b00
    # m3 = a00(b01 - b11)
    # m4 = a11 b11
    # m5 = (a00 + a10)b11
    # m6 = (a11 - a00)(b00 + b01)
    # m7 = (a01 - a11)(b10 + b11)
    
    # U coeffs (for A):
    # m1: a00:1, a11:1
    # m2: a10:1, a11:1
    # m3: a00:1
    # m4: a11:1
    # m5: a00:1, a10:1
    # m6: a11:1, a00:-1
    # m7: a01:1, a11:-1
    
    U_strassen = [
        [1, 0, 0, 1], # m1: a00, a01, a10, a11
        [0, 0, 1, 1], # m2
        [1, 0, 0, 0], # m3
        [0, 0, 0, 1], # m4
        [1, 0, 1, 0], # m5
        [-1, 0, 0, 1],# m6
        [0, 1, 0, -1] # m7
    ]
    
    # V coeffs (for B):
    # m1: b00:1, b11:1
    # m2: b00:1
    # m3: b01:1, b11:-1
    # m4: b11:1
    # m5: b11:1
    # m6: b00:1, b01:1
    # m7: b10:1, b11:1
    
    V_strassen = [
        [1, 0, 0, 1], # m1: b00, b01, b10, b11
        [1, 0, 0, 0], # m2
        [0, 1, 0, -1],# m3
        [0, 0, 0, 1], # m4
        [0, 0, 0, 1], # m5
        [1, 1, 0, 0], # m6
        [0, 0, 1, 1]  # m7
    ]
    
    # W coeffs (for C):
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    # W is indexed by c_row (0..1) and c_col (0..1) within the 2x2 block.
    # w_idx = i*8 + k. k = 2q + r.
    # So for a fixed q and r (0..1), we have 2 rows i (0..1).
    # We need to map the linear combination of m_s to W.
    # W[w_idx, col] is the coefficient of m_s for that output.
    
    W_strassen = [
        [1, 0, 0, 1],   # c00: m1, m2, m3, m4, m5, m6, m7 -> 1, 0, 0, 1, -1, 0, 1
        [0, 1, 1, 0],   # c01: 0, 0, 1, 0, 1, 0, 0
        [0, 1, 0, 1],   # c10: 0, 1, 0, 1, 0, 0, 0
        [1, -1, 1, 1]   # c11: 1, -1, 1, 0, 0, 1, 0
    ]
    
    # Wait, I need the full vector of coefficients for each m_s for each output.
    # Let's define W_coeffs[output_idx_in_2x2][s]
    
    W_coeffs = np.zeros((4, 7))
    # c00
    W_coeffs[0, 0] = 1  # m1
    W_coeffs[0, 3] = 1  # m4
    W_coeffs[0, 4] = -1 # m5
    W_coeffs[0, 6] = 1  # m7
    
    # c01
    W_coeffs[1, 2] = 1  # m3
    W_coeffs[1, 4] = 1  # m5
    
    # c10
    W_coeffs[2, 1] = 1  # m2
    W_coeffs[2, 3] = 1  # m4
    
    # c11
    W_coeffs[3, 0] = 1  # m1
    W_coeffs[3, 1] = -1 # m2
    W_coeffs[3, 2] = 1  # m3
    W_coeffs[3, 5] = 1  # m6
    
    for q in range(4):
        for s in range(7):
            # U part
            # u_idx = i*3 + j. i in 0..1, j in 0..1.
            # U_strassen[s] is [a00, a01, a10, a11]
            # j=0 corresponds to a00, a10? No.
            # A is 2x2. Rows i=0,1. Cols j=0,1.
            # a00 is i=0, j=0. u=0.
            # a01 is i=0, j=1. u=1.
            # a10 is i=1, j=0. u=3.
            # a11 is i=1, j=1. u=4.
            
            u_coeffs = U_strassen[s]
            # u_coeffs[0] is a00 -> u=0
            # u_coeffs[1] is a01 -> u=1
            # u_coeffs[2] is a10 -> u=3
            # u_coeffs[3] is a11 -> u=4
            
            U[0, col] = u_coeffs[0]
            U[1, col] = u_coeffs[1]
            U[3, col] = u_coeffs[2]
            U[4, col] = u_coeffs[3]
            
            # V part
            # v_idx = j'*8 + k. j' in 0..1. k = 2q + r.
            # V_strassen[s] is [b00, b01, b10, b11]
            # b00 is j'=0, k=2q. v = 0*8 + 2q = 2q.
            # b01 is j'=0, k=2q+1. v = 0*8 + 2q+1 = 2q+1.
            # b10 is j'=1, k=2q. v = 1*8 + 2q = 8+2q.
            # b11 is j'=1, k=2q+1. v = 1*8 + 2q+1 = 8+2q+1.
            
            v_coeffs = V_strassen[s]
            V[2*q, col] = v_coeffs[0]
            V[2*q+1, col] = v_coeffs[1]
            V[8+2*q, col] = v_coeffs[2]
            V[8+2*q+1, col] = v_coeffs[3]
            
            # W part
            # w_idx = i*8 + k. i in 0..1. k = 2q + r.
            # Output is 2x2.
            # c00: i=0, k=2q. w = 0*8 + 2q = 2q.
            # c01: i=0, k=2q+1. w = 0*8 + 2q+1 = 2q+1.
            # c10: i=1, k=2q. w = 1*8 + 2q = 8+2q.
            # c11: i=1, k=2q+1. w = 1*8 + 2q+1 = 8+2q+1.
            
            # W_coeffs[output_idx][s]
            # output_idx 0 -> c00 -> w=2q
            # output_idx 1 -> c01 -> w=2q+1
            # output_idx 2 -> c10 -> w=8+2q
            # output_idx 3 -> c11 -> w=8+2q+1
            
            W[2*q, col] = W_coeffs[0, s]
            W[2*q+1, col] = W_coeffs[1, s]
            W[8+2*q, col] = W_coeffs[2, s]
            W[8+2*q+1, col] = W_coeffs[3, s]
            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
