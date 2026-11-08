import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin (150s limit)
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42
    U = np.zeros((n * m, rank), dtype=int)  # 8 x 42
    V = np.zeros((m * p, rank), dtype=int)  # 24 x 42
    W = np.zeros((p * n, rank), dtype=int)  # 12 x 42
    
    def a_idx(i, j):
        return i * m + j
        
    def b_idx(i, j):
        return i * p + j
        
    def w_idx(i, j):
        return j * n + i

    col_offset = 0
    for group in [0, 1]:
        a_col_base = group * 2
        for block_col in [0, 1, 2]:
            b_row_base = group * 2
            b_col_base = block_col * 2
            c_col_base = block_col * 2
            
            # Term 1: (A00+A11)(B00+B11)
            U[a_idx(0, a_col_base + 0), col_offset] += 1
            U[a_idx(1, a_col_base + 1), col_offset] += 1
            V[b_idx(b_row_base + 0, b_col_base + 0), col_offset] += 1
            V[b_idx(b_row_base + 1, b_col_base + 1), col_offset] += 1
            W[w_idx(0, c_col_base + 0), col_offset] += 1
            W[w_idx(1, c_col_base + 1), col_offset] += 1
            col_offset += 1
            
            # Term 2: (A10+A11)B00
            U[a_idx(1, a_col_base + 0), col_offset] += 1
            U[a_idx(1, a_col_base + 1), col_offset] += 1
            V[b_idx(b_row_base + 0, b_col_base + 0), col_offset] += 1
            W[w_idx(1, c_col_base + 0), col_offset] += 1
            W[w_idx(1, c_col_base + 1), col_offset] -= 1
            col_offset += 1
            
            # Term 3: A00(B01-B11)
            U[a_idx(0, a_col_base + 0), col_offset] += 1
            V[b_idx(b_row_base + 0, b_col_base + 1), col_offset] += 1
            V[b_idx(b_row_base + 1, b_col_base + 1), col_offset] -= 1
            W[w_idx(0, c_col_base + 1), col_offset] += 1
            W[w_idx(1, c_col_base + 1), col_offset] += 1
            col_offset += 1
            
            # Term 4: A11(B10-B00)
            U[a_idx(1, a_col_base + 1), col_offset] += 1
            V[b_idx(b_row_base + 1, b_col_base + 0), col_offset] += 1
            V[b_idx(b_row_base + 0, b_col_base + 0), col_offset] -= 1
            W[w_idx(0, c_col_base + 0), col_offset] += 1
            W[w_idx(1, c_col_base + 0), col_offset] += 1
            col_offset += 1
            
            # Term 5: (A00+A01)B11
            U[a_idx(0, a_col_base + 0), col_offset] += 1
            U[a_idx(0, a_col_base + 1), col_offset] += 1
            V[b_idx(b_row_base + 1, b_col_base + 1), col_offset] += 1
            W[w_idx(0, c_col_base + 0), col_offset] -= 1
            W[w_idx(0, c_col_base + 1), col_offset] += 1
            col_offset += 1
            
            # Term 6: (A10-A00)(B00+B01)
            U[a_idx(1, a_col_base + 0), col_offset] += 1
            U[a_idx(0, a_col_base + 0), col_offset] -= 1
            V[b_idx(b_row_base + 0, b_col_base + 0), col_offset] += 1
            V[b_idx(b_row_base + 0, b_col_base + 1), col_offset] += 1
            W[w_idx(1, c_col_base + 1), col_offset] += 1
            col_offset += 1
            
            # Term 7: (A01-A11)(B10+B11)
            U[a_idx(0, a_col_base + 1), col_offset] += 1
            U[a_idx(1, a_col_base + 1), col_offset] -= 1
            V[b_idx(b_row_base + 1, b_col_base + 0), col_offset] += 1
            V[b_idx(b_row_base + 1, b_col_base + 1), col_offset] += 1
            W[w_idx(0, c_col_base + 0), col_offset] += 1
            col_offset += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
