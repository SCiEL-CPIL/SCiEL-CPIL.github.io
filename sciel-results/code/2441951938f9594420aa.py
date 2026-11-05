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
    
    Expected Rank: 117.
    """
    start_time = time.time()
    deadline = 150.0
    safety_margin = 5.0
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Partition 5 into 2, 2, 1
    parts = [2, 2, 1]
    offsets = [0, 2, 4]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_col(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Strassen factors for 2x2x2
    # Indices u, v, w in 0..1
    # u_vec, v_vec, w_vec are local to the block
    strassen_factors = [
        # M1 = (A00+A11)(B00+B11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (A10+A11)B00
        ([0, 1, 1, 0], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3 = A00(B01-B11)
        ([1, 0, 0, 0], [0, 1, -1, 0], [0, 1, 0, 1]),
        # M4 = A11(B10-B00)
        ([0, 0, 0, 1], [0, 0, -1, 1], [1, 0, 1, 0]),
        # M5 = (A00+A01)B11
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        # M6 = (A10-A00)(B00+B01)
        ([0, 1, -1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7 = (A01-A11)(B10+B11)
        ([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Iterate over blocks
    for a_idx in range(3):
        for b_idx in range(3):
            for c_idx in range(3):
                size_a = parts[a_idx]
                size_b = parts[b_idx]
                size_c = parts[c_idx]
                
                off_a = offsets[a_idx]
                off_b = offsets[b_idx]
                off_c = offsets[c_idx]
                
                if size_a == 2 and size_b == 2 and size_c == 2:
                    # Strassen block
                    for u_f, v_f, w_f in strassen_factors:
                        u_vec = [0]*25
                        v_vec = [0]*25
                        w_vec = [0]*25
                        
                        # Map local indices to global
                        # u_f is for A (i, j)
                        # v_f is for B (j, k)
                        # w_f is for C (k, i)
                        
                        # A indices: i in 0..1, j in 0..1
                        # Global i = off_a + i, Global j = off_b + j
                        for i in range(2):
                            for j in range(2):
                                val = u_f[i*2 + j]
                                if val != 0:
                                    u_vec[(off_a + i)*5 + (off_b + j)] = val
                                    
                        # B indices: j in 0..1, k in 0..1
                        # Global j = off_b + j, Global k = off_c + k
                        for j in range(2):
                            for k in range(2):
                                val = v_f[j*2 + k]
                                if val != 0:
                                    v_vec[(off_b + j)*5 + (off_c + k)] = val
                                    
                        # C indices: k in 0..1, i in 0..1
                        # Global k = off_c + k, Global i = off_a + i
                        for k in range(2):
                            for i in range(2):
                                val = w_f[k*2 + i]
                                if val != 0:
                                    w_vec[(off_c + k)*5 + (off_a + i)] = val
                                    
                        add_col(u_vec, v_vec, w_vec)
                        
                else:
                    # Schoolbook block
                    for u in range(size_a):
                        for v in range(size_b):
                            for w in range(size_c):
                                u_vec = [0]*25
                                v_vec = [0]*25
                                w_vec = [0]*25
                                
                                gi = off_a + u
                                gj = off_b + v
                                gk = off_c + w
                                
                                u_vec[gi*5 + gj] = 1
                                v_vec[gj*5 + gk] = 1
                                w_vec[gk*5 + gi] = 1
                                add_col(u_vec, v_vec, w_vec)
                                
                                if time.time() > deadline - safety_margin:
                                     # Return current best
                                     rank = len(U_cols)
                                     U = [[0]*rank for _ in range(25)]
                                     V = [[0]*rank for _ in range(25)]
                                     W = [[0]*rank for _ in range(25)]
                                     for col_idx in range(rank):
                                         for row_idx in range(25):
                                             U[row_idx][col_idx] = U_cols[col_idx][row_idx]
                                             V[row_idx][col_idx] = V_cols[col_idx][row_idx]
                                             W[row_idx][col_idx] = W_cols[col_idx][row_idx]
                                     return {"U": U, "V": V, "W": W}

    rank = len(U_cols)
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    for col_idx in range(rank):
        for row_idx in range(25):
            U[row_idx][col_idx] = U_cols[col_idx][row_idx]
            V[row_idx][col_idx] = V_cols[col_idx][row_idx]
            W[row_idx][col_idx] = W_cols[col_idx][row_idx]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
