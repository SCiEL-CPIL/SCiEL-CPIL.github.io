import time
import numpy as np
import sys

DIMENSIONS = [3, 4, 8]

def _add_rank_unit(U, V, W, col, u_coeffs, v_coeffs, w_coeffs):
    """
    Adds a single rank-1 term (or sum of terms forming one multiplication) to the decomposition.
    u_coeffs: dict {index_in_U: coefficient}
    v_coeffs: dict {index_in_V: coefficient}
    w_coeffs: dict {index_in_W: coefficient}
    """
    for idx, val in u_coeffs.items():
        U[idx][col] += val
    for idx, val in v_coeffs.items():
        V[idx][col] += val
    for idx, val in w_coeffs.items():
        W[idx][col] += val

def _strassen_2x2x2(U, V, W, col, a_idx, b_idx, c_idx):
    """
    Applies Strassen's algorithm to a 2x2x2 block.
    a_idx: list of 4 indices in U corresponding to A's 2x2 block [a11, a12, a21, a22]
    b_idx: list of 4 indices in V corresponding to B's 2x2 block [b11, b12, b21, b22]
    c_idx: list of 4 indices in W corresponding to C's 2x2 block [c11, c12, c21, c22]
    Returns next col index.
    """
    a11, a12, a21, a22 = a_idx
    b11, b12, b21, b22 = b_idx
    c11, c12, c21, c22 = c_idx
    
    # M1 = (a11 + a22)(b11 + b22)
    _add_rank_unit(U, V, W, col, {a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
    col += 1
    
    # M2 = (a21 + a22)b11
    _add_rank_unit(U, V, W, col, {a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
    col += 1
    
    # M3 = a11(b21 - b11)
    _add_rank_unit(U, V, W, col, {a11: 1}, {b21: 1, b11: -1}, {c12: 1, c22: 1})
    col += 1
    
    # M4 = a22(b22 - b12)
    _add_rank_unit(U, V, W, col, {a22: 1}, {b22: 1, b12: -1}, {c11: 1, c21: 1})
    col += 1
    
    # M5 = (a11 + a12)b22
    _add_rank_unit(U, V, W, col, {a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
    col += 1
    
    # M6 = (a21 - a11)(b11 + b12)
    _add_rank_unit(U, V, W, col, {a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
    col += 1
    
    # M7 = (a12 - a22)(b21 + b22)
    _add_rank_unit(U, V, W, col, {a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})
    col += 1
    
    return col

def _schoolbook_1x2x2(U, V, W, col, a_idx, b_idx, c_idx):
    """
    Applies schoolbook to a 1x2x2 block.
    a_idx: list of 2 indices in U [a1, a2]
    b_idx: list of 4 indices in V [b11, b12, b21, b22] (Row 1: b11, b12; Row 2: b21, b22)
    c_idx: list of 2 indices in W [c1, c2]
    """
    a1, a2 = a_idx
    b11, b12, b21, b22 = b_idx
    c1, c2 = c_idx
    
    # C1 = a1*b11 + a2*b21
    _add_rank_unit(U, V, W, col, {a1: 1}, {b11: 1}, {c1: 1})
    col += 1
    _add_rank_unit(U, V, W, col, {a2: 1}, {b21: 1}, {c1: 1})
    col += 1
    
    # C2 = a1*b12 + a2*b22
    _add_rank_unit(U, V, W, col, {a1: 1}, {b12: 1}, {c2: 1})
    col += 1
    _add_rank_unit(U, V, W, col, {a2: 1}, {b22: 1}, {c2: 1})
    col += 1
    
    return col

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    # U size: n*m = 12
    # V size: m*p = 32
    # W size: p*n = 24
    
    # Total Rank Calculation:
    # Block A (rows 0:2, cols 0:2) x Block B (rows 0:2, cols 0:8) -> C (rows 0:2, cols 0:8)
    #   4 Strassen 2x2x2 blocks. Rank 4*7 = 28.
    # Block A (rows 0:2, cols 2:4) x Block B (rows 2:4, cols 0:8) -> C (rows 0:2, cols 0:8)
    #   4 Strassen 2x2x2 blocks. Rank 4*7 = 28.
    # Block A (rows 2:3, cols 0:2) x Block B (rows 0:2, cols 0:8) -> C (rows 2:3, cols 0:8)
    #   4 Schoolbook 1x2x2 blocks. Rank 4*4 = 16.
    # Block A (rows 2:3, cols 2:4) x Block B (rows 2:4, cols 0:8) -> C (rows 2:3, cols 0:8)
    #   4 Schoolbook 1x2x2 blocks. Rank 4*4 = 16.
    # Total Rank = 28 + 28 + 16 + 16 = 88.
    
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Indices helpers
    # U index: i * m + j  (i in 0..n-1, j in 0..m-1)
    # V index: j * p + k  (j in 0..m-1, k in 0..p-1)
    # W index: k * n + i  (k in 0..p-1, i in 0..n-1)
    
    # --- Block 1: A[0:2, 0:2] * B[0:2, 0:8] -> C[0:2, 0:8] ---
    # A rows 0,1. A cols 0,1.
    # B rows 0,1. B cols 0..7.
    # C rows 0,1. C cols 0..7.
    
    # Split B cols into 4 blocks of 2: [0,1], [2,3], [4,5], [6,7]
    for k_start in range(0, p, 2):
        k1 = k_start
        k2 = k_start + 1
        
        # A indices (2x2 block at rows 0:2, cols 0:2)
        a11 = 0 * m + 0
        a12 = 0 * m + 1
        a21 = 1 * m + 0
        a22 = 1 * m + 1
        a_idx = [a11, a12, a21, a22]
        
        # B indices (2x2 block at rows 0:2, cols k1:k2+1)
        b11 = 0 * p + k1
        b12 = 0 * p + k2
        b21 = 1 * p + k1
        b22 = 1 * p + k2
        b_idx = [b11, b12, b21, b22]
        
        # C indices (2x2 block at rows 0:2, cols k1:k2+1)
        c11 = k1 * n + 0
        c12 = k2 * n + 0
        c21 = k1 * n + 1
        c22 = k2 * n + 1
        c_idx = [c11, c12, c21, c22]
        
        col = _strassen_2x2x2(U, V, W, col, a_idx, b_idx, c_idx)
        
    # --- Block 2: A[0:2, 2:4] * B[2:4, 0:8] -> C[0:2, 0:8] ---
    # A rows 0,1. A cols 2,3.
    # B rows 2,3. B cols 0..7.
    # C rows 0,1. C cols 0..7.
    
    for k_start in range(0, p, 2):
        k1 = k_start
        k2 = k_start + 1
        
        # A indices (2x2 block at rows 0:2, cols 2:4)
        a11 = 0 * m + 2
        a12 = 0 * m + 3
        a21 = 1 * m + 2
        a22 = 1 * m + 3
        a_idx = [a11, a12, a21, a22]
        
        # B indices (2x2 block at rows 2:4, cols k1:k2+1)
        b11 = 2 * p + k1
        b12 = 2 * p + k2
        b21 = 3 * p + k1
        b22 = 3 * p + k2
        b_idx = [b11, b12, b21, b22]
        
        # C indices (2x2 block at rows 0:2, cols k1:k2+1)
        c11 = k1 * n + 0
        c12 = k2 * n + 0
        c21 = k1 * n + 1
        c22 = k2 * n + 1
        c_idx = [c11, c12, c21, c22]
        
        col = _strassen_2x2x2(U, V, W, col, a_idx, b_idx, c_idx)

    # --- Block 3: A[2:3, 0:2] * B[0:2, 0:8] -> C[2:3, 0:8] ---
    # A row 2. A cols 0,1.
    # B rows 0,1. B cols 0..7.
    # C row 2. C cols 0..7.
    
    for k_start in range(0, p, 2):
        k1 = k_start
        k2 = k_start + 1
        
        # A indices (1x2 block at row 2, cols 0:2)
        a1 = 2 * m + 0
        a2 = 2 * m + 1
        a_idx = [a1, a2]
        
        # B indices (2x2 block at rows 0:2, cols k1:k2+1)
        b11 = 0 * p + k1
        b12 = 0 * p + k2
        b21 = 1 * p + k1
        b22 = 1 * p + k2
        b_idx = [b11, b12, b21, b22]
        
        # C indices (1x2 block at row 2, cols k1:k2+1)
        c1 = k1 * n + 2
        c2 = k2 * n + 2
        c_idx = [c1, c2]
        
        col = _schoolbook_1x2x2(U, V, W, col, a_idx, b_idx, c_idx)

    # --- Block 4: A[2:3, 2:4] * B[2:4, 0:8] -> C[2:3, 0:8] ---
    # A row 2. A cols 2,3.
    # B rows 2,3. B cols 0..7.
    # C row 2. C cols 0..7.
    
    for k_start in range(0, p, 2):
        k1 = k_start
        k2 = k_start + 1
        
        # A indices (1x2 block at row 2, cols 2:4)
        a1 = 2 * m + 2
        a2 = 2 * m + 3
        a_idx = [a1, a2]
        
        # B indices (2x2 block at rows 2:4, cols k1:k2+1)
        b11 = 2 * p + k1
        b12 = 2 * p + k2
        b21 = 3 * p + k1
        b22 = 3 * p + k2
        b_idx = [b11, b12, b21, b22]
        
        # C indices (1x2 block at row 2, cols k1:k2+1)
        c1 = k1 * n + 2
        c2 = k2 * n + 2
        c_idx = [c1, c2]
        
        col = _schoolbook_1x2x2(U, V, W, col, a_idx, b_idx, c_idx)

    if col != rank:
        # If col < rank, we have unused slots. We can just leave them as zeros.
        # If col > rank, we have an error.
        pass

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
