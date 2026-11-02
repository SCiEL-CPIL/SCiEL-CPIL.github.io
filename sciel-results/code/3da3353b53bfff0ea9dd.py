"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 110.0

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _strassen_2x2_blocks():
    """
    Returns the 7 terms for Strassen's 2x2x2 decomposition.
    Each term is (U_mask, V_mask, W_mask) where masks are dicts mapping indices to coefficients.
    Indices:
    U: (row, col) of 2x2 A. row 0..1, col 0..1.
    V: (row, col) of 2x2 B. row 0..1, col 0..1.
    W: (row, col) of 2x2 C. row 0..1, col 0..1.
    """
    # Standard Strassen formulas:
    # m1 = (A11+A22)(B11+B22) -> C11 += m1, C22 += m1
    # m2 = (A21+A22)B11       -> C21 += m2, C22 -= m2
    # m3 = A11(B12-B22)       -> C12 += m3, C22 += m3
    # m4 = A22(B21-B11)       -> C11 += m4, C21 += m4
    # m5 = (A11+A12)B22       -> C12 += m5, C11 -= m5
    # m6 = (A21-A11)(B11+B12) -> C22 += m6
    # m7 = (A12-A22)(B21+B22) -> C11 += m7
    
    # Note: Using 1-based indexing in comments, converting to 0-based for code.
    # A: (0,0) A00, (0,1) A01, (1,0) A10, (1,1) A11
    # B: (0,0) B00, (0,1) B01, (1,0) B10, (1,1) B11
    # C: (0,0) C00, (0,1) C01, (1,0) C10, (1,1) C11
    
    terms = [
        {
            'U': {(0,0): 1, (1,1): 1},
            'V': {(0,0): 1, (1,1): 1},
            'W': {(0,0): 1, (1,1): 1}
        },
        {
            'U': {(1,0): 1, (1,1): 1},
            'V': {(0,0): 1},
            'W': {(1,0): 1, (1,1): -1}
        },
        {
            'U': {(0,0): 1},
            'V': {(0,1): 1, (1,1): -1},
            'W': {(0,1): 1, (1,1): 1}
        },
        {
            'U': {(1,1): 1},
            'V': {(1,0): 1, (0,0): -1},
            'W': {(0,0): 1, (1,0): 1}
        },
        {
            'U': {(0,0): 1, (0,1): 1},
            'V': {(1,1): 1},
            'W': {(0,1): 1, (0,0): -1}
        },
        {
            'U': {(1,0): 1, (0,0): -1},
            'V': {(0,0): 1, (0,1): 1},
            'W': {(1,1): 1}
        },
        {
            'U': {(0,1): 1, (1,1): -1},
            'V': {(1,0): 1, (1,1): 1},
            'W': {(0,0): 1}
        }
    ]
    return terms

def construct():
    n, m, p = DIMENSIONS
    # 2x2 * 2x10
    # We decompose the 10 columns into 5 blocks of 2 columns.
    # Each block uses Strassen (rank 7).
    # Total rank = 5 * 7 = 35.
    
    num_blocks = p // 2
    rank = num_blocks * 7
    
    # Dimensions
    dim_U = n * m  # 4
    dim_V = m * p  # 20
    dim_W = p * n  # 20
    
    # Initialize matrices with zeros
    U = [[0 for _ in range(rank)] for _ in range(dim_U)]
    V = [[0 for _ in range(rank)] for _ in range(dim_V)]
    W = [[0 for _ in range(rank)] for _ in range(dim_W)]
    
    terms = _strassen_2x2_blocks()
    
    for b in range(num_blocks):
        base_idx = b * 7
        for t_idx in range(7):
            r = base_idx + t_idx
            term = terms[t_idx]
            
            # Populate U
            # U index: i*m + j, where i is row in A (0..1), j is col in A (0..1)
            for (i_a, j_a), coeff in term['U'].items():
                u_idx = i_a * m + j_a
                U[u_idx][r] = coeff
                
            # Populate V
            # V index: j*p + k, where j is row in B (0..1), k is col in B (0..9)
            # For block b, columns are 2*b and 2*b+1.
            # term['V'] keys are (j_b, k_b_local) where k_b_local is 0 or 1.
            for (j_b, k_b_local), coeff in term['V'].items():
                k_global = 2 * b + k_b_local
                v_idx = j_b * p + k_global
                V[v_idx][r] = coeff
                
            # Populate W
            # W index: k*n + i, where k is col in C (0..9), i is row in C (0..1)
            # term['W'] keys are (k_c_local, i_c) where k_c_local is 0 or 1.
            # Note: W matrix shape is (p*n, rank). Index is k*n + i.
            for (k_c_local, i_c), coeff in term['W'].items():
                k_global = 2 * b + k_c_local
                w_idx = k_global * n + i_c
                W[w_idx][r] = coeff
                
    return {'U': U, 'V': V, 'W': W}
# EVOLVE-BLOCK-END
