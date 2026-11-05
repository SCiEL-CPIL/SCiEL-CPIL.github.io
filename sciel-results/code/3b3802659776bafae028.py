"""Program for 5x5 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 5x5 matrix multiplication
    using a block-partition strategy with Strassen's algorithm for 2x2x2 sub-blocks.
    
    Strategy:
    1. Partition the dimension 5 into [2, 2, 1].
    2. Iterate over all 27 sub-blocks defined by these partitions.
    3. For 2x2x2 sub-blocks, use Strassen's rank-7 decomposition.
    4. For other sub-blocks, use the schoolbook rank (product of dimensions).
    5. Assemble the global U, V, W matrices.
    
    Expected Rank: 117 (vs 125 for pure schoolbook).
    """
    start_time = time.time()
    deadline = 150.0  # seconds, evaluator limit
    safety_margin = 5.0
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Partition 5 into 2, 2, 1
    parts = [2, 2, 1]
    offsets = [0, 2, 4]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a column
    def add_col(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Iterate over blocks
    for a_idx in range(3):
        for b_idx in range(3):
            for c_idx in range(3):
                size_a = parts[a_idx]
                size_b = parts[b_idx]
                size_c = parts[c_idx]
                
                # Global offsets
                off_a = offsets[a_idx]
                off_b = offsets[b_idx]
                off_c = offsets[c_idx]
                
                if size_a == 2 and size_b == 2 and size_c == 2:
                    # Strassen block for 2x2x2
                    # Local indices u,v,w in 0..1
                    # Global i = off_a + u, j = off_b + v, k = off_c + w
                    
                    # Strassen Factors (7 terms)
                    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[(off_a+0)*5 + (off_b+0)] = 1
                    u_vec[(off_a+1)*5 + (off_b+1)] = 1
                    v_vec[(off_b+0)*5 + (off_c+0)] = 1
                    v_vec[(off_b+1)*5 + (off_c+1)] = 1
                    w_vec[(off_c+0)*5 + (off_a+0)] = 1
                    w_vec[(off_c+1)*5 + (off_a+1)] = 1
                    add_col(u_vec, v_vec, w_vec)
                    
                    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[(off_a+1)*5 + (off_b+0)] = 1
                    u_vec[(off_a+1)*5 + (off_b+1)] = 1
                    v_vec[(off_b+0)*5 + (off_c+0)] = 1
                    w_vec[(off_c+0)*5 + (off_a+1)] = 1
                    w_vec[(off_c+1)*5 + (off_a+1)] = -1
                    add_col(u_vec, v_vec, w_vec)
                    
                    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[(off_a+0)*5 + (off_b+0)] = 1
                    v_vec[(off_b+0)*5 + (off_c+1)] = 1
                    v_vec[(off_b+1)*5 + (off_c+1)] = -1
                    w_vec[(off_c+1)*5 + (off_a+0)] = 1
                    w_vec[(off_c+1)*5 + (off_a+1)] = 1
                    add_col(u_vec, v_vec, w_vec)
                    
                    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[(off_a+1)*5 + (off_b+1)] = 1
                    v_vec[(off_b+1)*5 + (off_c+0)] = 1
                    v_vec[(off_b+0)*5 + (off_c+0)] = -1
                    w_vec[(off_c+0)*5 + (off_a+0)] = 1
                    w_vec[(off_c+0)*5 + (off_a+1)] = 1
                    add_col(u_vec, v_vec, w_vec)
                    
                    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[(off_a+0)*5 + (off_b+0)] = 1
                    u_vec[(off_a+0)*5 + (off_b+1)] = 1
                    v_vec[(off_b+1)*5 + (off_c+1)] = 1
                    w_vec[(off_c+0)*5 + (off_a+0)] = -1
                    w_vec[(off_c+1)*5 + (off_a+0)] = 1
                    add_col(u_vec, v_vec, w_vec)
                    
                    # M6 = (A10-A00)(B00+B01) -> C11 += M6
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[(off_a+1)*5 + (off_b+0)] = 1
                    u_vec[(off_a+0)*5 + (off_b+0)] = -1
                    v_vec[(off_b+0)*5 + (off_c+0)] = 1
                    v_vec[(off_b+0)*5 + (off_c+1)] = 1
                    w_vec[(off_c+1)*5 + (off_a+1)] = 1
                    add_col(u_vec, v_vec, w_vec)
                    
                    # M7 = (A01-A11)(B10+B11) -> C00 += M7
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[(off_a+0)*5 + (off_b+1)] = 1
                    u_vec[(off_a+1)*5 + (off_b+1)] = -1
                    v_vec[(off_b+1)*5 + (off_c+0)] = 1
                    v_vec[(off_b+1)*5 + (off_c+1)] = 1
                    w_vec[(off_c+0)*5 + (off_a+0)] = 1
                    add_col(u_vec, v_vec, w_vec)
                    
                else:
                    # Schoolbook block
                    for u in range(size_a):
                        for v in range(size_b):
                            for w in range(size_c):
                                u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                                gi = off_a + u
                                gj = off_b + v
                                gk = off_c + w
                                
                                u_vec[gi*5 + gj] = 1
                                v_vec[gj*5 + gk] = 1
                                w_vec[gk*5 + gi] = 1
                                add_col(u_vec, v_vec, w_vec)
                                
                                # Time check
                                if time.time() > deadline - safety_margin:
                                     # Return current best (partial) if timeout imminent, 
                                     # though this loop is fast enough to finish.
                                     pass

    # Convert lists of columns to lists of rows
    rank = len(U_cols)
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    for col_idx in range(rank):
        u_col = U_cols[col_idx]
        v_col = V_cols[col_idx]
        w_col = W_cols[col_idx]
        for row_idx in range(25):
            U[row_idx][col_idx] = u_col[row_idx]
            V[row_idx][col_idx] = v_col[row_idx]
            W[row_idx][col_idx] = w_col[row_idx]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
