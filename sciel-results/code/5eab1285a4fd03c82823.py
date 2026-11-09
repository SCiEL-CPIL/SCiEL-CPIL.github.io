import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    # Partition 5 = 2 + 2 + 1
    sizes = [2, 2, 1]
    n, m, p = DIMENSIONS  # all 5
    
    # Compute offsets
    offsets = [0, 2, 4]
    
    # Pre-compute block info
    block_info = []
    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                na, mb, pc = sizes[i], sizes[j], sizes[k]
                if na == 2 and mb == 2 and pc == 2:
                    b_rank = 7
                else:
                    b_rank = na * mb * pc
                block_info.append((i, j, k, na, mb, pc, b_rank, total_rank))
                total_rank += b_rank
    
    # Initialize matrices as NumPy arrays
    U = np.zeros((25, total_rank), dtype=int)
    V = np.zeros((25, total_rank), dtype=int)
    W = np.zeros((25, total_rank), dtype=int)
    
    for (i, j, k, na, mb, pc, b_rank, offset) in block_info:
        if na == 2 and mb == 2 and pc == 2:
            U_loc, V_loc, W_loc = strassen_2x2x2()
        else:
            U_loc, V_loc, W_loc = schoolbook(na, mb, pc)
        
        # Map to global
        # U index: global_i * 5 + global_j
        # V index: global_j * 5 + global_k
        # W index: global_k * 5 + global_i
        
        # Local indices: a in [0,na), b in [0,mb), c in [0,pc)
        # Global:
        # gi = offsets[i] + a
        # gj = offsets[j] + b
        # gk = offsets[k] + c
        
        for a in range(na):
            for b in range(mb):
                gi = offsets[i] + a
                gj = offsets[j] + b
                u_row = gi * 5 + gj
                # U_loc row: a * mb + b
                U[u_row, offset:offset+b_rank] = U_loc[a * mb + b, :]
        
        for b in range(mb):
            for c in range(pc):
                gj = offsets[j] + b
                gk = offsets[k] + c
                v_row = gj * 5 + gk
                # V_loc row: b * pc + c
                V[v_row, offset:offset+b_rank] = V_loc[b * pc + c, :]
        
        for c in range(pc):
            for a in range(na):
                gk = offsets[k] + c
                gi = offsets[i] + a
                w_row = gk * 5 + gi
                # W_loc row: c * na + a
                W[w_row, offset:offset+b_rank] = W_loc[c * na + a, :]
    
    return {"U": U, "V": V, "W": W}

def schoolbook(na, mb, pc):
    rank = na * mb * pc
    U = np.zeros((na * mb, rank), dtype=int)
    V = np.zeros((mb * pc, rank), dtype=int)
    W = np.zeros((pc * na, rank), dtype=int)
    col = 0
    for a in range(na):
        for b in range(mb):
            for c in range(pc):
                U[a * mb + b, col] = 1
                V[b * pc + c, col] = 1
                W[c * na + a, col] = 1
                col += 1
    return U, V, W

def strassen_2x2x2():
    rank = 7
    U = np.zeros((4, rank), dtype=int)
    V = np.zeros((4, rank), dtype=int)
    W = np.zeros((4, rank), dtype=int)
    
    # M0 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M1 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M2 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # M3 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # M4 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M5 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M6 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # C00 = M0 + M3 - M4 - M6 -> W[0] (k=0, i=0)
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = -1
    
    # C01 = M2 + M4 -> W[1] (k=0, i=1)
    W[1, 2] = 1; W[1, 4] = 1
    
    # C10 = M1 + M5 -> W[2] (k=1, i=0)
    W[2, 1] = 1; W[2, 5] = 1
    
    # C11 = M0 - M1 + M2 + M6 -> W[3] (k=1, i=1)
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 6] = 1
    
    return U, V, W
