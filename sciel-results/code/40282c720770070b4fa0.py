import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    deadline = time.time() + 145
    
    # Block decomposition parameters
    a_row_sizes = [2, 2]
    a_col_sizes = [2, 2, 1]
    b_row_sizes = [2, 2, 1]
    b_col_sizes = [2, 2, 1]
    c_row_sizes = [2, 2]
    c_col_sizes = [2, 2, 1]
    
    max_rank = n * m * p
    U = np.zeros((n * m, max_rank))
    V = np.zeros((m * p, max_rank))
    W = np.zeros((p * n, max_rank))
    rank = 0
    
    for i_blk in range(2):
        for j_blk in range(3):
            for k_blk in range(3):
                a_i_off = i_blk * 2
                a_j_off = sum(a_col_sizes[:j_blk])
                a_rows = a_row_sizes[i_blk]
                a_cols = a_col_sizes[j_blk]
                
                b_j_off = sum(b_row_sizes[:j_blk])
                b_k_off = sum(b_col_sizes[:k_blk])
                b_rows = b_row_sizes[j_blk]
                b_cols = b_col_sizes[k_blk]
                
                c_i_off = i_blk * 2
                c_k_off = sum(c_col_sizes[:k_blk])
                c_rows = c_row_sizes[i_blk]
                c_cols = c_col_sizes[k_blk]
                
                assert a_cols == b_rows
                
                if a_cols == 2 and c_rows == 2 and c_cols == 2:
                    # Strassen 2x2x2
                    strassen_terms = [
                        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
                        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
                        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
                        ({(1,1):1}, {(0,0):1, (1,0):-1}, {(0,0):1, (1,0):1}),
                        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
                        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
                        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
                    ]
                    for u_pat, v_pat, w_pat in strassen_terms:
                        r = rank
                        for (li, lj), val in u_pat.items():
                            U[(a_i_off + li) * m + (a_j_off + lj), r] = val
                        for (lj, lk), val in v_pat.items():
                            V[(b_j_off + lj) * p + (b_k_off + lk), r] = val
                        for (lk, li), val in w_pat.items():
                            W[(c_k_off + lk) * n + (c_i_off + li), r] = val
                        rank += 1
                else:
                    # Schoolbook
                    for li in range(c_rows):
                        for lj in range(a_cols):
                            for lk in range(c_cols):
                                r = rank
                                U[(a_i_off + li) * m + (a_j_off + lj), r] = 1
                                V[(b_j_off + lj) * p + (b_k_off + lk), r] = 1
                                W[(c_k_off + lk) * n + (c_i_off + li), r] = 1
                                rank += 1
    
    U = U[:, :rank]
    V = V[:, :rank]
    W = W[:, :rank]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
