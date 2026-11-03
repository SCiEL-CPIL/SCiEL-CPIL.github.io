"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    nm = n * m  # 16
    mp = m * p  # 28
    pn = p * n  # 28
    
    # Build target tensor
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    T_flat_a = T.reshape(nm, -1)  # (nm, mp*pn)
    T_flat_b = np.zeros((mp, nm * pn))
    for b in range(mp):
        T_flat_b[b] = T[:, b, :].reshape(-1)
    T_flat_c = np.zeros((pn, nm * mp))
    for c in range(pn):
        T_flat_c[c] = T[:, :, c].reshape(-1)
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p  # 112
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    def compute_error(U, V, W):
        err = np.zeros((nm, mp, pn), dtype=np.float64)
        for r in range(U.shape[1]):
            err += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.max(np.abs(T - err))
    
    def try_rank(target_rank, n_restarts=3, max_iter=150):
        nonlocal best_U, best_V, best_W, best_rank
        
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            # Structured initialization: start from schoolbook and randomize
            scale = 0.3
            U = np.random.randn(nm, target_rank) * scale
            V = np.random.randn(mp, target_rank) * scale
            W = np.random.randn(pn, target_rank) * scale
            
            prev_err = float('inf')
            stuck_count = 0
            
            for it in range(max_iter):
                if time.time() > deadline - 15:
                    break
                
                # Update U
                M_U = np.stack([np.outer(V[:, r], W[:, r]).reshape(-1) for r in range(target_rank)])
                try:
                    U_new = T_flat_a @ np.linalg.pinv(M_U)
                except:
                    break
                
                # Update V
                M_V = np.stack([np.outer(U_new[:, r], W[:, r]).reshape(-1) for r in range(target_rank)])
                try:
                    V_new = T_flat_b @ np.linalg.pinv(M_V)
                except:
                    break
                
                # Update W
                M_W = np.stack([np.outer(U_new[:, r], V_new[:, r]).reshape(-1) for r in range(target_rank)])
                try:
                    W_new = T_flat_c @ np.linalg.pinv(M_W)
                except:
                    break
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence every 5 iterations
                if it % 5 == 0 or it == max_iter - 1:
                    err = compute_error(U, V, W)
                    if err < 1e-7:
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_rank = target_rank
                        return True
                    if abs(err - prev_err) < 1e-10:
                        stuck_count += 1
                    else:
                        stuck_count = 0
                    prev_err = err
                    if stuck_count > 4:
                        break
            
            # Final check
            err = compute_error(U, V, W)
            if err < 1e-7:
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = target_rank
                return True
        
        return False
    
    # Try ranks from 100 down to 30
    # Focus on promising range
    candidate_ranks = list(range(100, 27, -1))
    
    for rank in candidate_ranks:
        if time.time() > deadline - 20:
            break
        success = try_rank(rank, n_restarts=2, max_iter=120)
        if success:
            break
    
    # If found better, use it
    if best_rank < rank_school:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
    else:
        U_out = U_school.tolist()
        V_out = V_school.tolist()
        W_out = W_school.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
