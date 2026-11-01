import numpy as np
import time

def construct():
    start_time = time.time()
    
    n, m, p = 2, 4, 7
    rank = 50
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_strassen_block(u_base, v_base, w_base, r_base):
        # M1
        U[u_base, r_base] += 1
        U[u_base + 3, r_base] += 1
        V[v_base, r_base] += 1
        V[v_base + 3, r_base] += 1
        W[w_base, r_base] += 1
        W[w_base + 3, r_base] += 1
        
        # M2
        U[u_base + 2, r_base + 1] += 1
        U[u_base + 3, r_base + 1] += 1
        V[v_base, r_base + 1] += 1
        W[w_base + 1, r_base + 1] += 1
        W[w_base + 3, r_base + 1] += 1
        
        # M3
        U[u_base, r_base + 2] += 1
        V[v_base + 1, r_base + 2] += 1
        V[v_base + 3, r_base + 2] += -1
        W[w_base + 2, r_base + 2] += 1
        W[w_base + 3, r_base + 2] += 1
        
        # M4
        U[u_base + 3, r_base + 3] += 1
        V[v_base + 2, r_base + 3] += 1
        V[v_base, r_base + 3] += -1
        W[w_base, r_base + 3] += 1
        W[w_base + 1, r_base + 3] += 1
        
        # M5
        U[u_base, r_base + 4] += 1
        U[u_base + 1, r_base + 4] += 1
        V[v_base + 3, r_base + 4] += 1
        W[w_base, r_base + 4] += -1
        W[w_base + 2, r_base + 4] += 1
        
        # M6
        U[u_base + 2, r_base + 5] += 1
        U[u_base, r_base + 5] += -1
        V[v_base, r_base + 5] += 1
        V[v_base + 1, r_base + 5] += 1
        W[w_base + 3, r_base + 5] += 1
        
        # M7
        U[u_base + 1, r_base + 6] += 1
        U[u_base + 3, r_base + 6] += -1
        V[v_base + 2, r_base + 6] += 1
        V[v_base + 3, r_base + 6] += 1
        W[w_base, r_base + 6] += 1

    def add_schoolbook_2x2x1_block(u_base, v_base, w_base, r_base):
        # C00 = A00 B00 + A01 B10
        # Term 1: A00 B00 -> C00
        U[u_base, r_base] += 1
        V[v_base, r_base] += 1
        W[w_base, r_base] += 1
        
        # Term 2: A01 B10 -> C00
        U[u_base + 1, r_base + 1] += 1
        V[v_base + 2, r_base + 1] += 1
        W[w_base, r_base + 1] += 1
        
        # C10 = A10 B00 + A11 B10
        # Term 3: A10 B00 -> C10
        U[u_base + 2, r_base + 2] += 1
        V[v_base, r_base + 2] += 1
        W[w_base + 1, r_base + 2] += 1
        
        # Term 4: A11 B10 -> C10
        U[u_base + 3, r_base + 3] += 1
        V[v_base + 2, r_base + 3] += 1
        W[w_base + 1, r_base + 3] += 1

    # Block 1: A_L (rows 0-1, cols 0-1 of A), B_L (rows 0-1 of B, all cols)
    # A_L indices in U: 0, 1, 2, 3
    # B_L indices in V: 0..6 (cols 0-6 of B, rows 0-1)
    # B is 4x7. Row 0 is 0..6, Row 1 is 7..13.
    # So B_L (rows 0-1) corresponds to V indices 0..13.
    # Wait, B_L is 2x7.
    # Split B_L into 2x2 (cols 0-1), 2x2 (cols 2-3), 2x2 (cols 4-5), 2x1 (col 6).
    
    # Sub-block 1.1: A_L, B_L[:, 0:2]
    # u_base = 0
    # v_base = 0 (B row 0 col 0 is V[0], B row 1 col 0 is V[7])
    # Wait, V index is row_B * p + col_B.
    # B_L[:, 0:2] uses B rows 0,1 and cols 0,1.
    # V indices: 
    # B[0,0] -> 0*7+0 = 0
    # B[0,1] -> 0*7+1 = 1
    # B[1,0] -> 1*7+0 = 7
    # B[1,1] -> 1*7+1 = 8
    # So for the 2x2 block, the V indices are not contiguous 0,1,2,3.
    # My helper `add_strassen_block` assumes contiguous indices v_base, v_base+1, v_base+2, v_base+3.
    # I need to adapt the helper or pass explicit indices.
    
    # Let's modify the helper to take lists of indices or just pass the base and stride?
    # Easier: Pass the specific indices for the 4 elements of A, B, C.
    
    def add_strassen_block_explicit(u_idx, v_idx, w_idx, r_base):
        # u_idx: [u00, u01, u10, u11]
        # v_idx: [v00, v01, v10, v11]
        # w_idx: [w00, w10, w01, w11] (Order: C00, C10, C01, C11)
        
        # M1
        U[u_idx[0], r_base] += 1
        U[u_idx[3], r_base] += 1
        V[v_idx[0], r_base] += 1
        V[v_idx[3], r_base] += 1
        W[w_idx[0], r_base] += 1
        W[w_idx[3], r_base] += 1
        
        # M2
        U[u_idx[2], r_base + 1] += 1
        U[u_idx[3], r_base + 1] += 1
        V[v_idx[0], r_base + 1] += 1
        W[w_idx[1], r_base + 1] += 1
        W[w_idx[3], r_base + 1] += 1
        
        # M3
        U[u_idx[0], r_base + 2] += 1
        V[v_idx[1], r_base + 2] += 1
        V[v_idx[3], r_base + 2] += -1
        W[w_idx[2], r_base + 2] += 1
        W[w_idx[3], r_base + 2] += 1
        
        # M4
        U[u_idx[3], r_base + 3] += 1
        V[v_idx[2], r_base + 3] += 1
        V[v_idx[0], r_base + 3] += -1
        W[w_idx[0], r_base + 3] += 1
        W[w_idx[1], r_base + 3] += 1
        
        # M5
        U[u_idx[0], r_base + 4] += 1
        U[u_idx[1], r_base + 4] += 1
        V[v_idx[3], r_base + 4] += 1
        W[w_idx[0], r_base + 4] += -1
        W[w_idx[2], r_base + 4] += 1
        
        # M6
        U[u_idx[2], r_base + 5] += 1
        U[u_idx[0], r_base + 5] += -1
        V[v_idx[0], r_base + 5] += 1
        V[v_idx[1], r_base + 5] += 1
        W[w_idx[3], r_base + 5] += 1
        
        # M7
        U[u_idx[1], r_base + 6] += 1
        U[u_idx[3], r_base + 6] += -1
        V[v_idx[2], r_base + 6] += 1
        V[v_idx[3], r_base + 6] += 1
        W[w_idx[0], r_base + 6] += 1

    def add_schoolbook_2x2x1_explicit(u_idx, v_idx, w_idx, r_base):
        # u_idx: [u00, u01, u10, u11]
        # v_idx: [v00, v10] (B is 2x1, so v00 is B[0,0], v10 is B[1,0])
        # w_idx: [w00, w10] (C is 2x1, so w00 is C[0,0], w10 is C[1,0])
        
        # Term 1: A00 B00 -> C00
        U[u_idx[0], r_base] += 1
        V[v_idx[0], r_base] += 1
        W[w_idx[0], r_base] += 1
        
        # Term 2: A01 B10 -> C00
        U[u_idx[1], r_base + 1] += 1
        V[v_idx[1], r_base + 1] += 1
        W[w_idx[0], r_base + 1] += 1
        
        # Term 3: A10 B00 -> C10
        U[u_idx[2], r_base + 2] += 1
        V[v_idx[0], r_base + 2] += 1
        W[w_idx[1], r_base + 2] += 1
        
        # Term 4: A11 B10 -> C10
        U[u_idx[3], r_base + 3] += 1
        V[v_idx[1], r_base + 3] += 1
        W[w_idx[1], r_base + 3] += 1

    r_curr = 0
    
    # Block 1: A_L (U indices 0,1,2,3)
    u_L = [0, 1, 2, 3]
    
    # B_L is rows 0,1 of B.
    # Cols 0-1: V indices [0, 1, 7, 8] (v00, v01, v10, v11)
    v_L1 = [0, 1, 7, 8]
    # W indices for C (cols 0-1, rows 0-1)
    # C[0,0] -> W[0], C[1,0] -> W[1], C[0,1] -> W[2], C[1,1] -> W[3]
    w_L1 = [0, 1, 2, 3]
    add_strassen_block_explicit(u_L, v_L1, w_L1, r_curr)
    r_curr += 7
    
    # Cols 2-3: V indices [2, 3, 9, 10]
    v_L2 = [2, 3, 9, 10]
    # W indices for C (cols 2-3)
    # C[0,2] -> W[2*2+0]=4, C[1,2] -> W[2*2+1]=5, C[0,3] -> W[3*2+0]=6, C[1,3] -> W[3*2+1]=7
    w_L2 = [4, 5, 6, 7]
    add_strassen_block_explicit(u_L, v_L2, w_L2, r_curr)
    r_curr += 7
    
    # Cols 4-5: V indices [4, 5, 11, 12]
    v_L3 = [4, 5, 11, 12]
    # W indices for C (cols 4-5)
    # C[0,4] -> W[4*2+0]=8, C[1,4] -> W[4*2+1]=9, C[0,5] -> W[5*2+0]=10, C[1,5] -> W[5*2+1]=11
    w_L3 = [8, 9, 10, 11]
    add_strassen_block_explicit(u_L, v_L3, w_L3, r_curr)
    r_curr += 7
    
    # Col 6: V indices [6, 13] (v00, v10)
    v_L4 = [6, 13]
    # W indices for C (col 6)
    # C[0,6] -> W[6*2+0]=12, C[1,6] -> W[6*2+1]=13
    w_L4 = [12, 13]
    add_schoolbook_2x2x1_explicit(u_L, v_L4, w_L4, r_curr)
    r_curr += 4
    
    # Block 2: A_R (U indices 4,5,6,7)
    u_R = [4, 5, 6, 7]
    
    # B_R is rows 2,3 of B.
    # V indices for B rows 2,3:
    # B[2,0] -> 2*7+0 = 14
    # B[3,0] -> 3*7+0 = 21
    
    # Cols 0-1: V indices [14, 15, 21, 22]
    v_R1 = [14, 15, 21, 22]
    # W indices same as L1: [0, 1, 2, 3]
    w_R1 = [0, 1, 2, 3]
    add_strassen_block_explicit(u_R, v_R1, w_R1, r_curr)
    r_curr += 7
    
    # Cols 2-3: V indices [16, 17, 23, 24]
    v_R2 = [16, 17, 23, 24]
    w_R2 = [4, 5, 6, 7]
    add_strassen_block_explicit(u_R, v_R2, w_R2, r_curr)
    r_curr += 7
    
    # Cols 4-5: V indices [18, 19, 25, 26]
    v_R3 = [18, 19, 25, 26]
    w_R3 = [8, 9, 10, 11]
    add_strassen_block_explicit(u_R, v_R3, w_R3, r_curr)
    r_curr += 7
    
    # Col 6: V indices [20, 27]
    v_R4 = [20, 27]
    w_R4 = [12, 13]
    add_schoolbook_2x2x1_explicit(u_R, v_R4, w_R4, r_curr)
    r_curr += 4
    
    # Check time
    if time.time() > start_time + 100:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
