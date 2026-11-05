import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 120  # seconds
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_schoolbook
    
    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    
    # Precompute reshaped tensors for ALS
    # T_reshaped_W: (n*m*m*p, p*n) where row index = (a,b), col = c
    T_W = np.zeros((n * m * m * p, p * n))
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                T_W[a * m * p + b, c] = T[a, b, c]
    
    # T_reshaped_U: (n*m, m*p*p*n) where row index = a, col index = (b,c)
    T_U = np.zeros((n * m, m * p * p * n))
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                T_U[a, b * p * n + c] = T[a, b, c]
    
    # T_reshaped_V: (m*p, n*m*p*n) where row index = b, col index = (a,c)
    T_V = np.zeros((m * p, n * m * p * n))
    for b in range(m * p):
        for a in range(n * m):
            for c in range(p * n):
                T_V[b, a * p * n + c] = T[a, b, c]
    
    # Try to find lower-rank decompositions using ALS
    for target_rank in range(35, 19, -1):
        if time.time() - start_time > time_limit - 30:
            break
        
        found = False
        for trial in range(30):
            if time.time() - start_time > time_limit - 30:
                break
            
            rng = np.random.RandomState(42 + trial * 7)
            U = rng.randn(n * m, target_rank)
            V = rng.randn(m * p, target_rank)
            W = rng.randn(p * n, target_rank)
            
            # ALS iterations
            for it in range(1000):
                if time.time() - start_time > time_limit - 30:
                    break
                
                # Update W: A @ W.T = T_W where A[(a,b), r] = U[a,r]*V[b,r]
                A = (U[:, :, None] * V[None, :, :]).reshape(n * m * m * p, target_rank)
                W_new, _, _, _ = np.linalg.lstsq(A, T_W, rcond=None)
                W = W_new.T  # (p*n, rank)
                
                # Update U: for each a, B[a] @ U[a,:] = T_U[a,:]
                # B[a, (b,c), r] = V[b,r]*W[c,r]
                B_all = np.einsum('br,cr->bcr', V, W).reshape(n * m, m * p * p * n, target_rank)
                U_new = np.zeros((n * m, target_rank))
                for a in range(n * m):
                    U_new[a], _, _, _ = np.linalg.lstsq(B_all[a], T_U[a], rcond=None)
                U = U_new
                
                # Update V: for each b, C[b] @ V[b,:] = T_V[b,:]
                # C[b, (a,c), r] = U[a,r]*W[c,r]
                C_all = np.einsum('ar,cr->acr', U, W).reshape(m * p, n * m * p * n, target_rank)
                V_new = np.zeros((m * p, target_rank))
                for b in range(m * p):
                    V_new[b], _, _, _ = np.linalg.lstsq(C_all[b], T_V[b], rcond=None)
                V = V_new
                
                # Check error
                E = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.sum((T - E) ** 2)
                if err < 1e-12:
                    found = True
                    break
                
                # Early termination if not improving
                if it > 100 and err > 1.0:
                    break
            
            if found:
                E = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.sum((T - E) ** 2)
                if err < 1e-10:
                    best_U = U
                    best_V = V
                    best_W = W
                    best_rank = target_rank
                    break
        
        if found and best_rank < rank_schoolbook:
            break
    
    if time.time() - start_time > time_limit - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
