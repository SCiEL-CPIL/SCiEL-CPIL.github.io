import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # We split p into p1=3 and p2=2
    p1 = 3
    p2 = 2
    
    # Target rank for M_{3,3,3}
    # Known best is 23.
    # We try to find a decomposition for M_{3,3,3}
    
    # Time limit guard
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    best_rank_333 = None
    best_decomp_333 = None
    
    # Try ranks from 23 to 27
    for r_try in range(23, 28):
        if time.time() > deadline:
            break
        decomp = _solve_tensor_decomp(3, 3, 3, r_try)
        if decomp is not None:
            best_rank_333 = r_try
            best_decomp_333 = decomp
            break
            
    if best_decomp_333 is None:
        # Fallback to schoolbook for 3,3,3
        best_rank_333 = 27
        best_decomp_333 = _schoolbook_decomp(3, 3, 3)
        
    # Schoolbook for 3,3,2
    decomp_332 = _schoolbook_decomp(3, 3, 2)
    rank_332 = 18
    
    total_rank = best_rank_333 + rank_332
    
    # Combine
    # U, V, W shapes: (n*m, r), (m*p, r), (p*n, r)
    # For 3,3,3: U1 (9, r1), V1 (9, r1), W1 (9, r1)
    # For 3,3,2: U2 (9, r2), V2 (6, r2), W2 (6, r2)
    
    U1, V1, W1 = best_decomp_333
    U2, V2, W2 = decomp_332
    
    # Pad V1 to size 15 (m*p = 3*5)
    # V1 has columns for k=0,1,2. We need to add zeros for k=3,4.
    # V1 shape (9, r1). Rows are j*p + k.
    # For p=3, rows are j*3 + k.
    # For p=5, rows are j*5 + k.
    # This mapping is tricky.
    # In the 3,3,3 decomp, V1[j*3+k, c] corresponds to B_{j,k} with k in 0..2.
    # In the final 3,3,5 decomp, V[j*5+k, c] should be non-zero for k in 0..2.
    # So we need to map index j*3+k to j*5+k.
    
    V1_full = np.zeros((m * p, best_rank_333))
    for j in range(m):
        for k in range(p1):
            V1_full[j * p + k] = V1[j * p1 + k]
            
    # W1 mapping:
    # W1[k*3+i, c] corresponds to C_{i,k} with k in 0..2.
    # In final, W[k*5+i, c].
    W1_full = np.zeros((p * n, best_rank_333))
    for k in range(p1):
        for i in range(n):
            W1_full[k * n + i] = W1[k * p1 + i]
            
    # U1 is already 9x r1, same as U for 3,3,5 (n*m=9).
    U1_full = U1
    
    # For 3,3,2 decomp:
    # U2 is 9x r2.
    # V2 is 6x r2 (j*2+k). k in 0..1.
    # In final, V[j*5+k, c] for k in 3..4.
    # So map j*2+k to j*5+(k+3).
    V2_full = np.zeros((m * p, rank_332))
    for j in range(m):
        for k in range(p2):
            V2_full[j * p + (k + p1)] = V2[j * p2 + k]
            
    # W2 is 6x r2 (k*3+i). k in 0..1.
    # In final, W[k*5+i, c] for k in 3..4.
    # Map k*3+i to (k+3)*5+i.
    W2_full = np.zeros((p * n, rank_332))
    for k in range(p2):
        for i in range(n):
            W2_full[(k + p1) * n + i] = W2[k * p1 + i]
            
    U2_full = U2
    
    # Concatenate
    U_final = np.hstack([U1_full, U2_full])
    V_final = np.hstack([V1_full, V2_full])
    W_final = np.hstack([W1_full, W2_full])
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

def _schoolbook_decomp(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _solve_tensor_decomp(n, m, p, rank):
    # Solve for M_{n,m,p}
    # Target tensor T
    # T[i,j, j',k, k',i'] = 1 if j=j', k=k', i=i'
    # We represent T as a 3-way tensor of size (n*m) x (m*p) x (p*n)
    size1 = n * m
    size2 = m * p
    size3 = p * n
    
    T = np.zeros((size1, size2, size3))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx1 = i * m + j
                idx2 = j * p + k
                idx3 = k * n + i
                T[idx1, idx2, idx3] = 1
                
    # ALS
    U = np.random.randn(size1, rank)
    V = np.random.randn(size2, rank)
    W = np.random.randn(size3, rank)
    
    for it in range(100):
        # Update U
        # T = sum_c U_c V_c^T W_c^T ?
        # T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
        # Fix V, W.
        # T_{a,:,:} = U_{a,:} @ (V * W)^T ?
        # Let M = V * W (element-wise product) of size (size2 * size3) x rank?
        # No.
        # T_{a, b, c} = sum_r U_{a,r} V_{b,r} W_{c,r}
        # Reshape T to (size1, size2*size3)
        T_mat = T.reshape(size1, size2 * size3)
        # M_{(b,c), r} = V_{b,r} W_{c,r}
        M = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(size2 * size3, rank)
        # T_mat = U @ M^T
        # U = T_mat @ M^+
        U = np.linalg.lstsq(M.T, T_mat.T, rcond=None)[0].T
        
        # Update V
        T_mat2 = T.reshape(size2, size1 * size3)
        # Permute T to (b, a, c)
        T_perm2 = np.transpose(T, (1, 0, 2)).reshape(size2, size1 * size3)
        M2 = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(size1 * size3, rank)
        V = np.linalg.lstsq(M2.T, T_perm2.T, rcond=None)[0].T
        
        # Update W
        T_mat3 = T.reshape(size3, size1 * size2)
        T_perm3 = np.transpose(T, (2, 0, 1)).reshape(size3, size1 * size3) # Wait, size1*size2
        T_perm3 = np.transpose(T, (2, 0, 1)).reshape(size3, size1 * size2)
        M3 = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(size1 * size2, rank)
        W = np.linalg.lstsq(M3.T, T_perm3.T, rcond=None)[0].T
        
        # Check error
        err = np.linalg.norm(T - (U[:, np.newaxis, np.newaxis] * V[np.newaxis, :, np.newaxis] * W[np.newaxis, np.newaxis, :]).reshape(size1, size2, size3))
        if err < 1e-6:
            # Check if close to integers
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            err_r = np.linalg.norm(T - (U_r[:, np.newaxis, np.newaxis] * V_r[np.newaxis, :, np.newaxis] * W_r[np.newaxis, np.newaxis, :]).reshape(size1, size2, size3))
            if err_r < 1e-6:
                return U_r, V_r, W_r
    return None

