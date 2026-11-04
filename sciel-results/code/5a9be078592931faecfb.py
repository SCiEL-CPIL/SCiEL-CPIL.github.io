import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition: rank = n*m*p = 96
    rank_school = n*m*p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank using a more careful alternating least squares
    # with rounding to integers/half-integers
    # The key insight: we need T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
    
    def check_decomposition(U, V, W):
        """Check if U, V, W exactly decompose T"""
        T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.allclose(T_approx, T, atol=1e-9)
    
    def round_to_half(x):
        """Round to nearest half-integer"""
        return np.round(x * 2) / 2
    
    # Try various ranks below schoolbook
    # Use alternating minimization with proper linear solves
    
    for target_rank in [88, 84, 80, 76, 72, 68, 64]:
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
            
            # Initialize with random combination of schoolbook columns
            # Pick target_rank columns from schoolbook
            idx = np.random.choice(rank_school, target_rank, replace=False)
            U = U_school[:, idx].copy()
            V = V_school[:, idx].copy()
            W = W_school[:, idx].copy()
            
            # Alternating least squares iterations
            for it in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Update U: T[i, :, :] = sum_c U[i,c] * (V[:,c] * W[:,c])
                # Flatten: T_flat[i, (j,k)] = sum_c U[i,c] * V[j,c] * W[k,c]
                # Build matrix M where M[c, j*p*n + k] = V[j,c] * W[k,c]
                M_VW = np.einsum('jc,kc->cjk', V, W).reshape(target_rank, m*p*p*n)
                
                T_flat = T.reshape(n*m, m*p*p*n)
                
                # Solve U @ M_VW.T = T_flat
                try:
                    # U = T_flat @ pinv(M_VW.T)
                    U_new, residuals, rank_m, sv = np.linalg.lstsq(M_VW.T, T_flat.T, rcond=None)
                    U = U_new.T
                except:
                    break
                
                # Update V: T[:, j, k] = sum_c U[i,c] * V[j,c] * W[k,c]
                # For fixed j: T[:, j, k] = sum_c V[j,c] * (U[:,c] * W[k,c])
                # Flatten: T_flat_v[j, (i,k)] = sum_c V[j,c] * U[i,c] * W[k,c]
                M_UW = np.einsum('ic,kc->cik', U, W).reshape(target_rank, n*m*p*n)
                
                # T[:, j, :] reshaped: for each j, we have (n*m, p*n)
                # T_flat_v[j, i*p*n + k] = T[i, j, k]
                T_flat_v = np.zeros((m*p, n*m*p*n))
                for j in range(m*p):
                    # j indexes into V, which is m*p
                    # But j in T is 0..m*p-1, and j in the tensor is j*p+k for the middle index
                    # Wait, I need to be more careful
                    pass
                
                # Let me redo this more carefully
                # T has indices (i*m+j, j*p+k, k*n+i) = 1
                # So T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
                # 
                # For the decomposition: T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
                #
                # Update U: fix V, W. For each a: T[a, :, :] = sum_r U[a,r] * (V[:,r] * W[:,r])
                # So T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # Flatten b,c: T_flat[a, b*(p*n) + c] = sum_r U[a,r] * (V[b,r] * W[c,r])
                # M_U[r, b*(p*n) + c] = V[b,r] * W[c,r]
                # Then T_flat = U @ M_U.T, so U = T_flat @ pinv(M_U.T)
                
                M_U = np.einsum('br,cr->bcr', V, W).reshape(m*p, p*n, target_rank).transpose(2, 0, 1).reshape(target_rank, m*p*p*n)
                T_flat_u = T.reshape(n*m, m*p*p*n)
                
                try:
                    U_new, _, _, _ = np.linalg.lstsq(M_U.T, T_flat_u.T, rcond=None)
                    U = U_new.T
                except:
                    break
                
                # Update V: fix U, W. For each b: T[:, b, :] = sum_r V[b,r] * (U[:,r] * W[:,r])
                # T[a, b, c] = sum_r V[b,r] * U[a,r] * W[c,r]
                # Flatten a,c: T_flat_v[b, a*(p*n) + c] = sum_r V[b,r] * (U[a,r] * W[c,r])
                M_V[r, a*(p*n) + c] = U[a,r] * W[c,r]
                M_V = np.einsum('ar,cr->acr', U, W).reshape(n*m, p*n, target_rank).transpose(2, 0, 1).reshape(target_rank, n*m*p*n)
                T_flat_v = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
                
                try:
                    V_new, _, _, _ = np.linalg.lstsq(M_V.T, T_flat_v.T, rcond=None)
                    V = V_new.T
                except:
                    break
                
                # Update W: fix U, V. For each c: T[:, :, c] = sum_r W[c,r] * (U[:,r] * V[:,r])
                # T[a, b, c] = sum_r W[c,r] * U[a,r] * V[b,r]
                # Flatten a,b: T_flat_w[c, a*(m*p) + b] = sum_r W[c,r] * (U[a,r] * V[b,r])
                M_W[r, a*(m*p) + b] = U[a,r] * V[b,r]
                M_W = np.einsum('ar,br->abr', U, V).reshape(n*m, m*p, target_rank).transpose(2, 0, 1).reshape(target_rank, n*m*m*p)
                T_flat_w = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p)
                
                try:
                    W_new, _, _, _ = np.linalg.lstsq(M_W.T, T_flat_w.T, rcond=None)
                    W = W_new.T
                except:
                    break
            
            # Check if current U, V, W (rounded) give exact decomposition
            U_r = round_to_half(U)
            V_r = round_to_half(V)
            W_r = round_to_half(W)
            
            if check_decomposition(U_r, V_r, W_r):
                found = True
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break
            
            # Also try without rounding
            if check_decomposition(U, V, W):
                found = True
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        
        if found:
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
