"""
Tensor decomposition for 4x4x4 matrix multiplication.
Uses the Kronecker product of the rank-7 Strassen decomposition for 2x2x2.
Rank = 7 * 7 = 49.
"""
import numpy as np
import time

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    of the tensor decomposition for 4x4 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Strassen decomposition for 2x2x2 (Rank 7)
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # Target: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k in {0,1}
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00 (k=0, i=0) -> W[0*2+0, c]
        [0, 0, 1, 0, 1, 0, 0],  # C01 (k=0, i=1) -> W[0*2+1, c]
        [0, 1, 0, 1, 0, 0, 0],  # C10 (k=1, i=0) -> W[1*2+0, c]
        [1, -1, 1, 0, 0, 1, 0]  # C11 (k=1, i=1) -> W[1*2+1, c]
    ], dtype=int)
    
    # Verify Strassen base
    valid_base = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += U2[i*2+j][c] * V2[j*2+k][c] * W2[k*2+i][c]
                if s != 1:
                    valid_base = False
                    break
            if not valid_base: break
        if not valid_base: break
        
    if not valid_base:
        return _schoolbook_4x4()

    # Construct Rank-49 decomposition via Kronecker Product
    rank = 49
    n, m, p = 4, 4, 4
    
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    # Iterate over all i, j, k to fill U, V, W
    # This approach ensures correct mapping of indices
    for i in range(4):
        for j in range(4):
            i1, i0 = divmod(i, 2)
            j1, j0 = divmod(j, 2)
            
            u_block_row = i1 * 2 + j1
            u_entry_row = i0 * 2 + j0
            
            u_idx = i * 4 + j
            
            for alpha in range(7):
                for beta in range(7):
                    col = alpha * 7 + beta
                    val_u = U2[u_block_row, alpha] * U2[u_entry_row, beta]
                    U[u_idx, col] = val_u
                    
            for k in range(4):
                k1, k0 = divmod(k, 2)
                
                v_block_row = j1 * 2 + k1
                v_entry_row = j0 * 2 + k0
                
                v_idx = j * 4 + k
                
                for alpha in range(7):
                    for beta in range(7):
                        col = alpha * 7 + beta
                        val_v = V2[v_block_row, alpha] * V2[v_entry_row, beta]
                        V[v_idx, col] = val_v
                        
                        w_block_row = k1 * 2 + i1
                        w_entry_row = k0 * 2 + i0
                        
                        w_idx = k * 4 + i
                        val_w = W2[w_block_row, alpha] * W2[w_entry_row, beta]
                        W[w_idx, col] = val_w
                        
    # Verify
    valid_49 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0
                for c in range(49):
                    s += U[i*4+j][c] * V[j*4+k][c] * W[k*4+i][c]
                if s != 1:
                    valid_49 = False
                    break
            if not valid_49: break
        if not valid_49: break
        
    if not valid_49:
        return _schoolbook_4x4()
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook_4x4():
    rank = 64
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4+j][col] = 1
                V[j*4+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
