import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 40
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for 150s limit
    
    # Build target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0
    
    def verify(U, V, W, tol=1e-10):
        """Check if U, V, W exactly decompose T."""
        n_m = n * m
        m_p = m * p
        p_n = p * n
        r = U.shape[1]
        # Compute sum_c U[a,c] * V[b,c] * W[d,c] for all a,b,d
        # Using einsum
        result = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.allclose(result, T, atol=tol, rtol=0)
    
    def alt_minimize(U, V, W, max_iter=500, tol=1e-12):
        """Alternating minimization for tensor decomposition."""
        n_m = n * m
        m_p = m * p
        p_n = p * n
        r = U.shape[1]
        
        T_flat = T.reshape(n_m, m_p * p_n)
        
        prev_err = np.inf
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Fix V and W, solve for U
            # M[c, (b,d)] = V[b,c] * W[d,c], shape (r, m_p*p_n)
            M_vw = np.outer(V.T, W.T).reshape(r, m_p * p_n)  # M[c, b*p_n+d] = V[b,c]*W[d,c]
            # For each a: U[a,:] @ M_vw = T_flat[a,:]
            # Solve: U[a,:] = T_flat[a,:] @ M_vw^+ (pseudoinverse)
            try:
                U_new = np.linalg.lstsq(M_vw, T_flat.T, rcond=None)[0].T
            except:
                break
            
            # Fix U and W, solve for V
            # N[c, (a,d)] = U[a,c] * W[d,c], shape (r, n_m*p_n)
            M_uw = np.outer(U_new.T, W.T).reshape(r, n_m * p_n)
            T_flat_v = T.reshape(n_m, m_p, p_n).transpose(1, 0, 2).reshape(m_p, n_m * p_n)
            try:
                V_new = np.linalg.lstsq(M_uw, T_flat_v.T, rcond=None)[0].T
            except:
                break
            
            # Fix U and V, solve for W
            # P[c, (a,b)] = U[a,c] * V[b,c], shape (r, n_m*m_p)
            M_uv = np.outer(U_new.T, V_new.T).reshape(r, n_m * m_p)
            T_flat_w = T.reshape(n_m, m_p, p_n).transpose(2, 0, 1).reshape(p_n, n_m * m_p)
            try:
                W_new = np.linalg.lstsq(M_uv, T_flat_w.T, rcond=None)[0].T
            except:
                break
            
            # Check convergence
            result = np.einsum('ac,bc,dc->abd', U_new, V_new, W_new)
            err = np.linalg.norm(result - T)
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
            U, V, W = U_new, V_new, W_new
        
        return U, V, W, err
    
    # Try ranks from schoolbook down
    best_result = None
    best_rank = rank_schoolbook
    
    # First, create the schoolbook solution as fallback
    U_fb = np.zeros((n * m, rank_schoolbook))
    V_fb = np.zeros((m * p, rank_schoolbook))
    W_fb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
    
    # Try to find lower rank decompositions
    np.random.seed(42)
    for rank in range(rank_schoolbook - 1, max(20, rank_schoolbook - 15), -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        found = False
        for trial in range(5):
            if time.time() > deadline:
                break
            
            U_init = np.random.randn(n * m, rank)
            V_init = np.random.randn(m * p, rank)
            W_init = np.random.randn(p * n, rank)
            
            U_opt, V_opt, W_opt, err = alt_minimize(U_init, V_init, W_init, max_iter=200, tol=1e-12)
            
            if err < 1e-10:
                if verify(U_opt, V_opt, W_opt):
                    best_result = (U_opt, V_opt, W_opt)
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
    
    if best_result is not None:
        U, V, W = best_result
    else:
        U, V, W = U_fb, V_fb, W_fb
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

