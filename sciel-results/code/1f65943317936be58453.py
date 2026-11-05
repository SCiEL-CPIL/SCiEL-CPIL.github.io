"""
Tensor decomposition for 3x4 * 4x8 matrix multiplication.
Uses Strassen's algorithm for 2x2 blocks where possible.
"""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4*4x8 matrix multiplication.
    
    Strategy:
    - Partition A (3x4) into A1 (2x4) and A2 (1x4)
    - Partition B (4x8) into 4 blocks of 4x2
    - For A1*B: use Strassen on 2x2 blocks (8 blocks of 2x2*2x2, each rank 7 = 56 total)
    - For A2*B: schoolbook 1x4*4x8, rank 32
    Total rank: 56 + 32 = 88
    """
    start_time = time.time()
    DEADLINE = 120.0  # seconds, safety margin
    
    n, m, p = DIMENSIONS
    
    terms = []  # List of (u_indices, u_values, v_indices, v_values, w_indices, w_values)
    
    def add_strassen_block(i0, j0, k0):
        """
        Add Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        
        Strassen's algorithm for 2x2*2x2:
        Let A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        
        M1 = (a11 + a22) * (b11 + b22)
        M2 = (a21 + a22) * b11
        M3 = a11 * (b12 - b22)
        M4 = a22 * (b21 - b11)
        M5 = (a11 + a12) * b22
        M6 = (a21 - a11) * (b11 + b12)
        M7 = (a12 - a22) * (b21 + b22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        """
        # Map local indices to global indices
        a11_u = (i0) * m + j0
        a12_u = (i0) * m + (j0 + 1)
        a21_u = (i0 + 1) * m + j0
        a22_u = (i0 + 1) * m + (j0 + 1)
        
        b11_v = j0 * p + k0
        b12_v = j0 * p + (k0 + 1)
        b21_v = (j0 + 1) * p + k0
        b22_v = (j0 + 1) * p + (k0 + 1)
        
        c11_w = k0 * n + i0
        c12_w = (k0 + 1) * n + i0
        c21_w = k0 * n + (i0 + 1)
        c22_w = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a11 + a22) * (b11 + b22)
        terms.append(([a11_u, a22_u], [1.0, 1.0], [b11_v, b22_v], [1.0, 1.0], [c11_w, c22_w], [1.0, 1.0]))
        
        # M2 = (a21 + a22) * b11
        terms.append(([a21_u, a22_u], [1.0, 1.0], [b11_v], [1.0], [c21_w, c22_w], [1.0, -1.0]))
        
        # M3 = a11 * (b12 - b22)
        terms.append(([a11_u], [1.0], [b12_v, b22_v], [1.0, -1.0], [c12_w, c22_w], [1.0, 1.0]))
        
        # M4 = a22 * (b21 - b11)
        terms.append(([a22_u], [1.0], [b21_v, b11_v], [1.0, -1.0], [c11_w, c21_w], [1.0, 1.0]))
        
        # M5 = (a11 + a12) * b22
        terms.append(([a11_u, a12_u], [1.0, 1.0], [b22_v], [1.0], [c11_w, c12_w], [-1.0, 1.0]))
        
        # M6 = (a21 - a11) * (b11 + b12)
        terms.append(([a21_u, a11_u], [1.0, -1.0], [b11_v, b12_v], [1.0, 1.0], [c22_w], [1.0]))
        
        # M7 = (a12 - a22) * (b21 + b22)
        terms.append(([a12_u, a22_u], [1.0, -1.0], [b21_v, b22_v], [1.0, 1.0], [c11_w], [1.0]))
    
    def add_schoolbook_1x4_4x8(i_row, j0, k0):
        """
        Add schoolbook decomposition for A[i_row, j0:j0+4] * B[j0:j0+4, k0:k0+8]
        """
        for j in range(4):
            for k in range(8):
                u_idx = i_row * m + (j0 + j)
                v_idx = (j0 + j) * p + (k0 + k)
                w_idx = (k0 + k) * n + i_row
                terms.append(([u_idx], [1.0], [v_idx], [1.0], [w_idx], [1.0]))
    
    # 8 Strassen blocks for A[0:2, :] * B[:, :]
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    add_strassen_block(0, 0, 6)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    add_strassen_block(0, 2, 6)
    
    # Schoolbook for A[2, :] * B[:, :]
    add_schoolbook_1x4_4x8(2, 0, 0)
    
    rank = len(terms)
    
    # Build final matrices using numpy for speed
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        for idx, val in zip(u_idx, u_val):
            U[idx, r] = val
        for idx, val in zip(v_idx, v_val):
            V[idx, r] = val
        for idx, val in zip(w_idx, w_val):
            W[idx, r] = val
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
