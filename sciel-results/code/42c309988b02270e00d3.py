import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook fallback
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin before 150s kill
    
    # Target tensor for 2x3x9
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def als_step(U, V, W, T, r):
        """One ALS iteration for tensor decomposition."""
        nm, mp, pn = T.shape
        # Update U
        for a in range(nm):
            M = (V[:, None, :] * W[None, :, :]).reshape(mp * pn, r)
            target = T[a, :, :].flatten()
            try:
                U[a, :], _, _, _ = np.linalg.lstsq(M.T, target, rcond=None)
            except:
                pass
        # Update V
        for b in range(mp):
            M = (U[:, None, :] * W[None, :, :]).reshape(nm * pn, r)
            target = T[:, b, :].reshape(nm, pn).flatten()
            try:
                V[b, :], _, _, _ = np.linalg.lstsq(M.T, target, rcond=None)
            except:
                pass
        # Update W
        for c in range(pn):
            M = (U[:, None, :] * V[None, :, :]).reshape(nm * mp, r)
            target = T[:, :, c].reshape(nm, mp).flatten()
            try:
                W[c, :], _, _, _ = np.linalg.lstsq(M.T, target, rcond=None)
            except:
                pass
        return U, V, W
    
    def compute_error(U, V, W, T):
        """Compute ||sum_r U_r V_r W_r - T||^2"""
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T_approx - T) ** 2)
    
    def verify_decomposition(U, V, W, T, tol=1e-8):
        """Verify and correct a decomposition."""
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.sum((T_approx - T) ** 2)
        if err < tol:
            return True, U, V, W
        return False, U, V, W
    
    def find_decomposition(T, nm, mp, pn, r, max_iters=200, n_restarts=10, seed_base=42):
        """Find rank-r decomposition using ALS with multiple restarts."""
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for restart in range(n_restarts):
            if time.time() - start_time > deadline:
                break
                
            seed = seed_base + restart * 1000
            np.random.seed(seed)
            
            # Structured initialization: use schoolbook-like patterns
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            
            # Initialize with some random components
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.5
            W = np.random.randn(pn, r) * 0.5
            
            prev_error = compute_error(U, V, W, T)
            
            for it in range(max_iters):
                if time.time() - start_time > deadline:
                    break
                    
                U, V, W = als_step(U, V, W, T, r)
                error = compute_error(U, V, W, T)
                
                if error < 1e-12:
                    return U, V, W, error
                
                if error > 1e6:
                    break
                if abs(error - prev_error) < 1e-16:
                    break
                prev_error = error
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        return best_U, best_V, best_W, best_error
    
    # Try to find low-rank decomposition for 2x3x3
    # Known: schoolbook is 18, best known is 16
    n_s, m_s, p_s = 2, 3, 3
    T_s = np.zeros((n_s * m_s, m_s * p_s, p_s * n_s))
    for i in range(n_s):
        for j in range(m_s):
            for k in range(p_s):
                T_s[i * m_s + j, j * p_s + k, k * n_s + i] = 1.0
    
    nm_s, mp_s, pn_s = n_s * m_s, m_s * p_s, p_s * n_s
    
    # Try ranks from low to high for 2x3x3
    for r_s in [12, 13, 14, 15, 16, 17, 18]:
        if time.time() - start_time > deadline - 30:
            break
            
        U_s, V_s, W_s, err = find_decomposition(T_s, nm_s, mp_s, pn_s, r_s, 
                                                   max_iters=150, n_restarts=20)
        
        if U_s is not None and err < 1e-8:
            # Found rank-r_s decomposition for 2x3x3
            # Construct 2x3x9 decomposition with rank 3*r_s
            rank_total = 3 * r_s
            if rank_total < best_rank:
                # U: (nm, 3*r_s) - repeat U_s three times horizontally
                U_new = np.hstack([U_s, U_s, U_s])
                
                # V: (mp, 3*r_s) - block diagonal
                V_new = np.zeros((m * p, 3 * r_s))
                for b in range(3):
                    for j in range(m_s):
                        for k in range(p_s):
                            row_new = j * p + (b * p_s + k)
                            row_s = j * p_s + k
                            V_new[row_new, b * r_s : (b + 1) * r_s] = V_s[row_s, :]
                
                # W: (pn, 3*r_s) - block diagonal
                W_new = np.zeros((p * n, 3 * r_s))
                for b in range(3):
                    for k in range(p_s):
                        for i in range(n_s):
                            row_new = (b * p_s + k) * n + i
                            row_s = k * n_s + i
                            W_new[row_new, b * r_s : (b + 1) * r_s] = W_s[row_s, :]
                
                # Verify
                T_approx = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
                verify_err = np.sum((T_approx - T) ** 2)
                
                if verify_err < 1e-8:
                    best_U = U_new.tolist()
                    best_V = V_new.tolist()
                    best_W = W_new.tolist()
                    best_rank = rank_total
                    break  # Found a good one, stop searching
    
    # If we didn't find a better one via 2x3x3 expansion, try direct search for 2x3x9
    if best_rank == rank_schoolbook and time.time() - start_time < deadline - 20:
        for r in range(45, rank_schoolbook + 1, 3):
            if time.time() - start_time > deadline:
                break
            if r >= best_rank:
                continue
                
            U, V, W, err = find_decomposition(T, n * m, m * p, p * n, r,
                                               max_iters=80, n_restarts=8)
            if U is not None and err < 1e-8:
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                best_rank = r
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
