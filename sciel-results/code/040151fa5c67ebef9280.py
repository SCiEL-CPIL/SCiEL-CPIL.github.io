import numpy as np
import time

def construct():
    """
    Rank-28 decomposition for [2,4,4] matrix multiplication.
    
    Uses 4 independent 2x2x2 Strassen blocks (rank 7 each).
    Each block handles a 2x2 sub-product.
    """
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    def u_idx(i, j): return i * m + j
    def v_idx(i, j): return i * p + j
    def w_idx(i, j): return j * n + i
    
    # Four 2x2 blocks: (a_row, a_col, b_row, b_col, c_col)
    # A is 2x4, B is 4x4, C is 4x2
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # We split into 4 blocks of 2x2x2:
    # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 1: A[0:2, 2:4] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 2: A[0:2, 0:2] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Block 3: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    
    blocks = [
        (0, 0, 0, 0, 0),  # A rows 0-1, cols 0-1; B rows 0-1, cols 0-1; C cols 0-1
        (0, 2, 0, 0, 0),  # A rows 0-1, cols 2-3; B rows 0-1, cols 0-1; C cols 0-1
        (0, 0, 2, 0, 0),  # A rows 0-1, cols 0-1; B rows 2-3, cols 0-1; C cols 0-1
        (0, 2, 2, 0, 0),  # A rows 0-1, cols 2-3; B rows 2-3, cols 0-1; C cols 0-1
    ]
    
    # Strassen terms for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M2 + M3 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M4 + M5 + M6
    
    # local indices: 0=A00, 1=A01, 2=A10, 3=A11
    #                0=B00, 1=B01, 2=B10, 3=B11
    #                0=C00, 1=C01, 2=C10, 3=C11
    
    strassen_terms = [
        # (U_terms, V_terms, W_terms)
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, -1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(2, 1), (0, -1)], [(1, 1), (0, 1)]),
        # M4
        ([(3, 1)], [(1, 1), (3, -1)], [(2, 1), (3, -1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (3, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    current_rank = 0
    
    for (a_r, a_c, b_r, b_c, c_c) in blocks:
        u_map = [
            u_idx(a_r, a_c),
            u_idx(a_r, a_c + 1),
            u_idx(a_r + 1, a_c),
            u_idx(a_r + 1, a_c + 1)
        ]
        
        v_map = [
            v_idx(b_r, b_c),
            v_idx(b_r, b_c + 1),
            v_idx(b_r + 1, b_c),
            v_idx(b_r + 1, b_c + 1)
        ]
        
        w_map = [
            w_idx(c_c, 0),
            w_idx(c_c, 1),
            w_idx(c_c + 1, 0),
            w_idx(c_c + 1, 1)
        ]
        
        for (u_terms, v_terms, w_terms) in strassen_terms:
            for local_idx, coeff in u_terms:
                U[u_map[local_idx], current_rank] += coeff
            for local_idx, coeff in v_terms:
                V[v_map[local_idx], current_rank] += coeff
            for local_idx, coeff in w_terms:
                W[w_map[local_idx], current_rank] += coeff
            current_rank += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
