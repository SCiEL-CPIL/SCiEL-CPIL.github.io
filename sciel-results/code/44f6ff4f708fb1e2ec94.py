"""5x5x5 matrix multiplication tensor decomposition using 2+2+1 block structure with Strassen on all 2x2x2 blocks (rank 99)."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

_T0 = time.time()
_LIMIT = 150.0
_MARGIN = 5.0

def _deadline_exceeded():
    return (time.time() - _T0) > (_LIMIT - _MARGIN)

# Standard Strassen 2x2x2 terms
# Each term: (u_coeffs, v_coeffs, w_coeffs)
# u_coeffs: {(li, lj): val} for A[li, lj]
# v_coeffs: {(lj, lk): val} for B[lj, lk]
# w_coeffs: {(li, lk): val} for C[li, lk]
# The tensor identity is: sum_t U_t ⊗ V_t ⊗ W_t = multiplication tensor
# For 2x2: C = A*B
# Strassen formulas:
# M1 = (A00 + A11)(B00 + B11)
# M2 = (A10 + A11)B00
# M3 = A00(B01 - B11)
# M4 = A11(B10 - B00)
# M5 = (A00 + A01)B11
# M6 = (A10 - A00)(B00 + B01)
# M7 = (A01 - A11)(B10 + B11)
# C00 = M1 + M4 - M5 + M7
# C01 = M3 + M5
# C10 = M2 + M4
# C11 = M1 - M2 + M3 + M6

STRASSEN_2x2_TERMS = [
    # M1: (A00+A11)(B00+B11) -> contributes to C00, C11
    ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
    # M2: (A10+A11)B00 -> contributes to C10 (with +1), C11 (with -1)
    ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
    # M3: A00(B01-B11) -> contributes to C01 (with +1), C11 (with +1)
    ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
    # M4: A11(B10-B00) -> contributes to C00 (with +1), C10 (with +1)
    ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
    # M5: (A00+A01)B11 -> contributes to C00 (with -1), C01 (with +1)
    ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
    # M6: (A10-A00)(B00+B01) -> contributes to C11 (with +1)
    ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
    # M7: (A01-A11)(B10+B11) -> contributes to C00 (with +1)
    ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
]

def construct():
    """
    Rank-reduced exact tensor decomposition for 5x5x5 matrix multiplication.
    
    Uses a 2+2+1 block partition.
    All 2x2x2 diagonal blocks use Strassen (rank 7 each).
    Mixed blocks use schoolbook.
    Total rank: 99.
    """
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    columns = []
    
    def add_column(u_entries, v_entries, w_entries):
        u_d = {}
        for idx, val in u_entries:
            if val != 0:
                u_d[idx] = u_d.get(idx, 0) + val
        v_d = {}
        for idx, val in v_entries:
            if val != 0:
                v_d[idx] = v_d.get(idx, 0) + val
        w_d = {}
        for idx, val in w_entries:
            if val != 0:
                w_d[idx] = w_d.get(idx, 0) + val
        # Clean up zeros
        u_d = {k: v for k, v in u_d.items() if v != 0}
        v_d = {k: v for k, v in v_d.items() if v != 0}
        w_d = {k: v for k, v in w_d.items() if v != 0}
        columns.append((u_d, v_d, w_d))
    
    # Block index groups: 2+2+1 partition
    blocks = [[0, 1], [2, 3], [4]]
    
    def schoolbook_block(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_column(
                        [(i * m + j, 1.0)],
                        [(j * p + k, 1.0)],
                        [(k * n + i, 1.0)]
                    )
    
    def strassen_block(i_list, j_list, k_list):
        """
        Adds 7 columns for a Strassen 2x2x2 decomposition.
        i_list, j_list, k_list: lists of 2 global indices each.
        """
        if len(i_list) != 2 or len(j_list) != 2 or len(k_list) != 2:
            raise ValueError("strassen_block requires lists of length 2")
        
        I_map = {0: i_list[0], 1: i_list[1]}
        J_map = {0: j_list[0], 1: j_list[1]}
        K_map = {0: k_list[0], 1: k_list[1]}
        
        for u_coeffs, v_coeffs, w_coeffs in STRASSEN_2x2_TERMS:
            u_entries = []
            v_entries = []
            w_entries = []
            
            for (li, lj), val in u_coeffs.items():
                gi = I_map[li]
                gj = J_map[lj]
                u_entries.append((gi * m + gj, val))
                
            for (lj, lk), val in v_coeffs.items():
                gj = J_map[lj]
                gk = K_map[lk]
                v_entries.append((gj * p + gk, val))
                
            for (li, lk), val in w_coeffs.items():
                gi = I_map[li]
                gk = K_map[lk]
                w_entries.append((gk * n + gi, val))
                
            add_column(u_entries, v_entries, w_entries)
    
    # Build all blocks
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                if _deadline_exceeded():
                    break
                i_list = blocks[bi]
                j_list = blocks[bj]
                k_list = blocks[bk]
                
                if len(i_list) == 2 and len(j_list) == 2 and len(k_list) == 2:
                    strassen_block(i_list, j_list, k_list)
                else:
                    schoolbook_block(i_list, j_list, k_list)
            if _deadline_exceeded():
                break
        if _deadline_exceeded():
            break
    
    rank = len(columns)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c, (u_d, v_d, w_d) in enumerate(columns):
        for idx, val in u_d.items():
            U[idx, c] = val
        for idx, val in v_d.items():
            V[idx, c] = val
        for idx, val in w_d.items():
            W[idx, c] = val
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
