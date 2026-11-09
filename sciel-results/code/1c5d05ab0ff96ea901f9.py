"""Matrix multiplication 3,4,8 using rank-88 decomposition with improved rank reduction."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Initial decomposition: rank-88
    rank = 88
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Part 1: i=0, schoolbook for 1x4x8 (rank 32)
    for j in range(4):
        for k in range(8):
            U[0 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 0, col] = 1
            col += 1
    
    # Part 2: i=1,2, Strassen on 2x2 blocks (rank 56)
    s_U = [
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0], 
        [0, 1, 0, -1]  
    ]
    s_V = [
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1], 
        [-1, 0, 1, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ]
    s_W = [
        [1, 0, 0, 1], 
        [0, 0, 1, -1], 
        [0, 1, 0, 1], 
        [1, 0, 1, 0], 
        [-1, 1, 0, 0], 
        [0, 0, 0, 1], 
        [1, 0, 0, 0]  
    ]
    
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            for t in range(7):
                for l in range(4):
                    if s_U[t][l] != 0:
                        li = l // 2
                        lj = l % 2
                        gi = 1 + li
                        gj = j_start + lj
                        U[gi * m + gj, col] += s_U[t][l]
                    
                    if s_V[t][l] != 0:
                        lj = l // 2
                        lk = l % 2
                        gj = j_start + lj
                        gk = k_start + lk
                        V[gj * p + gk, col] += s_V[t][l]
                    
                    if s_W[t][l] != 0:
                        li = l // 2
                        lk = l % 2
                        gi = 1 + li
                        gk = k_start + lk
                        W[gk * n + gi, col] += s_W[t][l]
                col += 1
    
    # Verify initial decomposition
    def verify(U_mat, V_mat, W_mat):
        recon = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        return np.allclose(recon, target, atol=1e-10)
    
    if not verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Rank reduction: try to find linear dependencies
    # For each column r, check if it can be removed by adjusting others
    # We'll use a greedy approach with actual linear algebra
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try removing each column and see if we can compensate
    for r_remove in range(rank):
        if time.time() > deadline - 5:
            break
        
        new_rank = U.shape[1] - 1
        if new_rank >= best_rank:
            continue
        
        U_new = np.delete(U, r_remove, axis=1)
        V_new = np.delete(V, r_remove, axis=1)
        W_new = np.delete(W, r_remove, axis=1)
        
        # Check if the remaining columns still span the target
        if verify(U_new, V_new, W_new):
            best_rank = new_rank
            best_U, best_V, best_W = U_new, V_new, W_new
            U, V, W = U_new, V_new, W_new
    
    # Try a second pass
    for r_remove in range(U.shape[1]):
        if time.time() > deadline - 3:
            break
        
        new_rank = U.shape[1] - 1
        if new_rank >= best_rank:
            continue
        
        U_new = np.delete(U, r_remove, axis=1)
        V_new = np.delete(V, r_remove, axis=1)
        W_new = np.delete(W, r_remove, axis=1)
        
        if verify(U_new, V_new, W_new):
            best_rank = new_rank
            best_U, best_V, best_W = U_new, V_new, W_new
            U, V, W = U_new, V_new, W_new
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
