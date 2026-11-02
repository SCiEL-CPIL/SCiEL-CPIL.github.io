import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Build the target tensor M of shape (n*m, m*p, p*n)
    M = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                M[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n*m*p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    deadline = time.time() + 100  # 100s deadline with margin
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Try ranks from 14 to 72
    for rank in range(14, rank_schoolbook + 1):
        if time.time() > deadline:
            break
        
        found = False
        for restart in range(10):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Alternating least squares
            for it in range(30):
                if time.time() > deadline:
                    break
                
                # Update U
                VW = np.kron(V, W)  # (m*p*p*n, rank)
                M_mat = M.reshape(n*m, m*p*p*n)
                U = np.linalg.lstsq(VW, M_mat, rcond=None)[0]
                
                # Update V
                UW = np.kron(U, W)  # (n*m*p*n, rank)
                M_mat2 = M.reshape(m*p, n*m*p*n)
                V = np.linalg.lstsq(UW.T, M_mat2.T, rcond=None)[0].T
                
                # Update W
                UV = np.kron(U, V)  # (n*m*m*p, rank)
                M_mat3 = M.reshape(p*n, n*m*m*p)
                W = np.linalg.lstsq(UV.T, M_mat3.T, rcond=None)[0].T
            
            # Round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify
            M_recon = np.einsum('it,jt,kt->ijk', U_rounded, V_rounded, W_rounded)
            if np.allclose(M_recon, M, atol=1e-10):
                best_rank = rank
                best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                found = True
                break
        
        if found:
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
