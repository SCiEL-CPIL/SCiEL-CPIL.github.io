"""
Tensor decomposition for 2x4 by 4x4 matrix multiplication.
Uses a structured rank-28 decomposition via Strassen on 2x2 blocks.
"""
import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    """
    Returns a rank-28 exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x4 matrix.
    
    Strategy: Split A (2x4) into A_L (2x2) and A_R (2x2).
    Split B (4x4) into four 2x2 blocks: TL, BL, TR, BR.
    C_L = A_L * B_TL + A_R * B_BL
    C_R = A_L * B_TR + A_R * B_BR
    
    Use Strassen (rank 7) for each 2x2x2 product.
    Total rank = 4 * 7 = 28.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen coefficients for 2x2x2
    # Each term: (coeff_A, coeff_B, coeff_C)
    # where coeff_X is a dict mapping (row, col) -> coefficient
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10+A11)B00 -> contributes to C10, C11 (with sign)
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A00(B01-B11) -> contributes to C01, C11 (with sign)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A11(B10-B00) -> contributes to C00, C10 (with sign)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A00+A01)B11 -> contributes to C00, C01 (with sign)
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (A10-A00)(B00+B01) -> contributes to C10
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,0): 1}),
        # M7 = (A01-A11)(B10+B11) -> contributes to C01
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,1): 1}),
    ]
    
    # Define the four 2x2x2 subproblems
    # Each subproblem: (A_block, B_block, C_block)
    # A_block: which 2x2 block of A (L=cols 0-1, R=cols 2-3)
    # B_block: which 2x2 block of B (TL, BL, TR, BR)
    # C_block: which 2x2 block of C (L=cols 0-1, R=cols 2-3)
    
    subproblems = [
        ('L', 'TL', 'L'),  # A_L * B_TL -> C_L
        ('R', 'BL', 'L'),  # A_R * B_BL -> C_L
        ('L', 'TR', 'R'),  # A_L * B_TR -> C_R
        ('R', 'BR', 'R'),  # A_R * B_BR -> C_R
    ]
    
    # Mapping functions
    def map_u(a_block, r, c):
        """Map 2x2 A-block coordinates to global U index."""
        if a_block == 'L':
            return r * 4 + c  # cols 0-1
        else:  # R
            return r * 4 + (c + 2)  # cols 2-3
    
    def map_v(b_block, r, c):
        """Map 2x2 B-block coordinates to global V index."""
        if b_block == 'TL':
            return r * 4 + c  # rows 0-1, cols 0-1
        elif b_block == 'BL':
            return (r + 2) * 4 + c  # rows 2-3, cols 0-1
        elif b_block == 'TR':
            return r * 4 + (c + 2)  # rows 0-1, cols 2-3
        else:  # BR
            return (r + 2) * 4 + (c + 2)  # rows 2-3, cols 2-3
    
    def map_w(c_block, r, c):
        """Map 2x2 C-block coordinates to global W index.
        C is p x n = 4 x 2, so index = c * 2 + r where c is col, r is row."""
        if c_block == 'L':
            return c * 2 + r  # cols 0-1
        else:  # R
            return (c + 2) * 2 + r  # cols 2-3
    
    current_rank = 0
    for a_block, b_block, c_block in subproblems:
        for term in strassen_terms:
            coeff_A, coeff_B, coeff_C = term
            
            for (r, c), val in coeff_A.items():
                u_idx = map_u(a_block, r, c)
                U[u_idx][current_rank] += val
            
            for (r, c), val in coeff_B.items():
                v_idx = map_v(b_block, r, c)
                V[v_idx][current_rank] += val
            
            for (r, c), val in coeff_C.items():
                w_idx = map_w(c_block, r, c)
                W[w_idx][current_rank] += val
            
            current_rank += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
