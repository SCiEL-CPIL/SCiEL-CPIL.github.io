import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = 2, 3, 6
    nm = n * m  # 6
    mp = m * p  # 18
    np_ = n * p  # 12
    
    # Build the tensor T
    T = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook
    rank_sb = nm * mp * np_  # 36
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((np_, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try ALS for various ranks
    for rank in range(18, best_rank):
        if time.time() > deadline - 10:
            break
        
        found = False
        for trial in range(3):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(rank * 1000 + trial)
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(np_, rank)
            
            for it in range(10):
                if time.time() > deadline - 3:
                    break
                
                # Update U
                for a in range(nm):
                    A_mat = (V[:, None, :] * W[None, :, :]).reshape(mp*np_, rank)
                    b_vec = T[a].flatten()
                    U[a, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                # Update V
                for b in range(mp):
                    A_mat = (U[:, None, :] * W[None, :, :]).reshape(nm*np_, rank)
                    b_vec = T[:, b, :].flatten()
                    V[b, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                # Update W
                for c in range(np_):
                    A_mat = (U[:, None, :] * V[None, :, :]).reshape(nm*mp, rank)
                    b_vec = T[:, :, c].flatten()
                    W[c, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                # Check error
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                error = np.linalg.norm(recon - T)
                
                if error < 1e-10:
                    found = True
                    break
            
            if found:
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
        
        if best_rank == rank:
            break
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
