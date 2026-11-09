import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    total_rank = 44
    mat_U = np.zeros((n * m, total_rank), dtype=int)
    mat_V = np.zeros((m * p, total_rank), dtype=int)
    mat_W = np.zeros((p * n, total_rank), dtype=int)
    
    current_col = 0
    
    def strassen_2x2x2(u_off_i, u_off_j, v_off_j, v_off_k, w_off_k, w_off_i):
        nonlocal current_col
        
        # M1 = (A00 + A11) * (B00 + B11)
        mat_U[(u_off_i+0)*m + (u_off_j+0), current_col] = 1
        mat_U[(u_off_i+1)*m + (u_off_j+1), current_col] = 1
        mat_V[(v_off_j+0)*p + (v_off_k+0), current_col] = 1
        mat_V[(v_off_j+1)*p + (v_off_k+1), current_col] = 1
        mat_W[(w_off_k+0)*n + (w_off_i+0), current_col] = 1
        mat_W[(w_off_k+1)*n + (w_off_i+1), current_col] = 1
        current_col += 1
        
        # M2 = (A10 + A11) * B00
        mat_U[(u_off_i+1)*m + (u_off_j+0), current_col] = 1
        mat_U[(u_off_i+1)*m + (u_off_j+1), current_col] = 1
        mat_V[(v_off_j+0)*p + (v_off_k+0), current_col] = 1
        mat_W[(w_off_k+0)*n + (w_off_i+1), current_col] = 1
        mat_W[(w_off_k+1)*n + (w_off_i+1), current_col] = -1
        current_col += 1
        
        # M3 = A00 * (B01 - B11)
        mat_U[(u_off_i+0)*m + (u_off_j+0), current_col] = 1
        mat_V[(v_off_j+0)*p + (v_off_k+1), current_col] = 1
        mat_V[(v_off_j+1)*p + (v_off_k+1), current_col] = -1
        mat_W[(w_off_k+1)*n + (w_off_i+0), current_col] = 1
        mat_W[(w_off_k+1)*n + (w_off_i+1), current_col] = 1
        current_col += 1
        
        # M4 = A11 * (B10 - B00)
        mat_U[(u_off_i+1)*m + (u_off_j+1), current_col] = 1
        mat_V[(v_off_j+1)*p + (v_off_k+0), current_col] = 1
        mat_V[(v_off_j+0)*p + (v_off_k+0), current_col] = -1
        mat_W[(w_off_k+0)*n + (w_off_i+0), current_col] = 1
        mat_W[(w_off_k+0)*n + (w_off_i+1), current_col] = 1
        current_col += 1
        
        # M5 = (A00 + A01) * B11
        mat_U[(u_off_i+0)*m + (u_off_j+0), current_col] = 1
        mat_U[(u_off_i+0)*m + (u_off_j+1), current_col] = 1
        mat_V[(v_off_j+1)*p + (v_off_k+1), current_col] = 1
        mat_W[(w_off_k+0)*n + (w_off_i+0), current_col] = -1
        mat_W[(w_off_k+1)*n + (w_off_i+0), current_col] = 1
        current_col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        mat_U[(u_off_i+1)*m + (u_off_j+0), current_col] = 1
        mat_U[(u_off_i+0)*m + (u_off_j+0), current_col] = -1
        mat_V[(v_off_j+0)*p + (v_off_k+0), current_col] = 1
        mat_V[(v_off_j+0)*p + (v_off_k+1), current_col] = 1
        mat_W[(w_off_k+1)*n + (w_off_i+1), current_col] = 1
        current_col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        mat_U[(u_off_i+0)*m + (u_off_j+1), current_col] = 1
        mat_U[(u_off_i+1)*m + (u_off_j+1), current_col] = -1
        mat_V[(v_off_j+1)*p + (v_off_k+0), current_col] = 1
        mat_V[(v_off_j+1)*p + (v_off_k+1), current_col] = 1
        mat_W[(w_off_k+0)*n + (w_off_i+0), current_col] = 1
        current_col += 1

    def schoolbook_1x2x2(u_off_i, u_off_j, v_off_j, v_off_k, w_off_k, w_off_i):
        nonlocal current_col
        
        # C00 = A00 B00 + A01 B10
        # Term 1
        mat_U[(u_off_i+0)*m + (u_off_j+0), current_col] = 1
        mat_V[(v_off_j+0)*p + (v_off_k+0), current_col] = 1
        mat_W[(w_off_k+0)*n + (w_off_i+0), current_col] = 1
        current_col += 1
        
        # Term 2
        mat_U[(u_off_i+0)*m + (u_off_j+1), current_col] = 1
        mat_V[(v_off_j+1)*p + (v_off_k+0), current_col] = 1
        mat_W[(w_off_k+0)*n + (w_off_i+0), current_col] = 1
        current_col += 1
        
        # C01 = A00 B01 + A01 B11
        # Term 3
        mat_U[(u_off_i+0)*m + (u_off_j+0), current_col] = 1
        mat_V[(v_off_j+0)*p + (v_off_k+1), current_col] = 1
        mat_W[(w_off_k+1)*n + (w_off_i+0), current_col] = 1
        current_col += 1
        
        # Term 4
        mat_U[(u_off_i+0)*m + (u_off_j+1), current_col] = 1
        mat_V[(v_off_j+1)*p + (v_off_k+1), current_col] = 1
        mat_W[(w_off_k+1)*n + (w_off_i+0), current_col] = 1
        current_col += 1

    # Top 2 rows of A (indices 0, 1)
    # Split B into 2x2 blocks (4 blocks)
    # Block (0,0): i in [0,1], j in [0,1], k in [0,1]
    strassen_2x2x2(0, 0, 0, 0, 0, 0)
    # Block (0,1): i in [0,1], j in [2,3], k in [0,1]
    strassen_2x2x2(0, 2, 2, 0, 0, 0)
    # Block (1,0): i in [0,1], j in [0,1], k in [2,3]
    strassen_2x2x2(0, 0, 0, 2, 2, 0)
    # Block (1,1): i in [0,1], j in [2,3], k in [2,3]
    strassen_2x2x2(0, 2, 2, 2, 2, 0)

    # Bottom 1 row of A (index 2)
    # Split B into 2x2 blocks (4 blocks)
    # Block (0,0): i=2, j in [0,1], k in [0,1]
    schoolbook_1x2x2(2, 0, 0, 0, 0, 0)
    # Block (0,1): i=2, j in [2,3], k in [0,1]
    schoolbook_1x2x2(2, 2, 2, 0, 0, 0)
    # Block (1,0): i=2, j in [0,1], k in [2,3]
    schoolbook_1x2x2(2, 0, 0, 2, 2, 0)
    # Block (1,1): i=2, j in [2,3], k in [2,3]
    schoolbook_1x2x2(2, 2, 2, 2, 2, 0)

    return {"U": mat_U.tolist(), "V": mat_V.tolist(), "W": mat_W.tolist()}
# EVOLVE-BLOCK-END
