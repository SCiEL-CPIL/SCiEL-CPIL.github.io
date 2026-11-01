import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 7]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U,V,W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            lhs = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            rhs = 0.0
                            for r in range(rank):
                                rhs += U[i*m+j, r] * V[jp*p+kp, r] * W[kp*n+ip, r]
                            if abs(lhs - rhs) > tol:
                                return False
    return True

def _make_schoolbook(n, m, p):
    rank = n * m * p
    n_m = n * m
    m_p = m * p
    p_n = p * n
    U = np.zeros((n_m, rank))
    V = np.zeros((m_p, rank))
    W = np.zeros((p_n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _solve_subtensor(n, m, p, target_rank, deadline, max_trials=20):
    """Try to find a decomposition of rank target_rank for T_{n,m,p}."""
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    T = np.zeros((n_m, m_p, p_n))
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            if i == ip and j == jp and k == kp:
                                a = i*m + j
                                b = jp*p + k
                                c = kp*n + ip
                                T[a, b, c] = 1.0
    T_flat = T.flatten()
    
    def residuals(params):
        U_r = params[:n_m*target_rank].reshape(n_m, target_rank)
        V_r = params[n_m*target_rank:n_m*target_rank + m_p*target_rank].reshape(m_p, target_rank)
        W_r = params[n_m*target_rank + m_p*target_rank:].reshape(p_n, target_rank)
        T_calc = np.einsum('ak,bk,ck->abc', U_r, V_r, W_r)
        return T_calc.flatten() - T_flat
    
    for trial in range(max_trials):
        if time.time() > deadline:
            break
        
        np.random.seed(42 + trial)
        U = np.random.randn(n_m, target_rank) * 0.3
        V = np.random.randn(m_p, target_rank) * 0.3
        W = np.random.randn(p_n, target_rank) * 0.3
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=2000, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            
            if result.success or np.linalg.norm(result.fun) < 1e-8:
                U_new = result.x[:n_m*target_rank].reshape(n_m, target_rank)
                V_new = result.x[n_m*target_rank:n_m*target_rank + m_p*target_rank].reshape(m_p, target_rank)
                W_new = result.x[n_m*target_rank + m_p*target_rank:].reshape(p_n, target_rank)
                
                if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                    return U_new, V_new, W_new
        except Exception:
            pass
    
    return None, None, None

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    n, m, p = DIMENSIONS
    
    # Direct sum approach: T_{2,2,7} = T_{2,2,4} ⊕ T_{2,2,3}
    # Try to find rank-11 for (2,2,4) and rank-8 for (2,2,3)
    # Total rank would be 19
    
    p1, p2 = 4, 3
    rank1, rank2 = 11, 8
    
    U1, V1, W1 = _solve_subtensor(2, 2, p1, rank1, deadline - 30)
    
    if time.time() > deadline - 25:
        # Fallback to schoolbook if running out of time
        U, V, W = _make_schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U2, V2, W2 = _solve_subtensor(2, 2, p2, rank2, deadline - 10)
    
    if U1 is not None and U2 is not None:
        # Combine: U_total = hstack(U1, U2)
        # V_total: rows 0-7 from V1, rows 8-13 from V2
        # W_total: rows 0-7 from W1, rows 8-13 from W2
        
        total_rank = rank1 + rank2
        n_m = n * m
        m_p = m * p
        p_n = p * n
        
        U_total = np.hstack([U1, U2])
        
        V_total = np.zeros((m_p, total_rank))
        V_total[:2*p1, :] = 0
        V_total[2*p1:2*p1 + 2*p2, :] = 0
        V_total[:2*p1, :rank1] = V1
        V_total[2*p1:2*p1 + 2*p2, rank1:] = V2
        
        W_total = np.zeros((p_n, total_rank))
        W_total[:p1*n, :] = 0
        W_total[p1*n:p1*n + p2*n, :] = 0
        W_total[:p1*n, :rank1] = W1
        W_total[p1*n:p1*n + p2*n, rank1:] = W2
        
        if _verify_decomposition(U_total, V_total, W_total, n, m, p):
            return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
    
    # If direct sum didn't work, try full search with lower rank
    # Try ranks from 19 up to 28
    for r in range(19, n*m*p + 1):
        if time.time() > deadline - 10:
            break
        
        n_m = n * m
        m_p = m * p
        p_n = p * n
        
        T = np.zeros((n_m, m_p, p_n))
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                if i == ip and j == jp and k == kp:
                                    a = i*m + j
                                    b = jp*p + k
                                    c = kp*n + ip
                                    T[a, b, c] = 1.0
        T_flat = T.flatten()
        
        def residuals_full(params):
            U_r = params[:n_m*r].reshape(n_m, r)
            V_r = params[n_m*r:n_m*r + m_p*r].reshape(m_p, r)
            W_r = params[n_m*r + m_p*r:].reshape(p_n, r)
            T_calc = np.einsum('ak,bk,ck->abc', U_r, V_r, W_r)
            return T_calc.flatten() - T_flat
        
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(100 + r * 10 + trial)
            U = np.random.randn(n_m, r) * 0.5
            V = np.random.randn(m_p, r) * 0.5
            W = np.random.randn(p_n, r) * 0.5
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = least_squares(residuals_full, x0, method='lm', max_nfev=500, ftol=1e-8, xtol=1e-8, gtol=1e-8)
                
                if result.success or np.linalg.norm(result.fun) < 1e-8:
                    U_new = result.x[:n_m*r].reshape(n_m, r)
                    V_new = result.x[n_m*r:n_m*r + m_p*r].reshape(m_p, r)
                    W_new = result.x[n_m*r + m_p*r:].reshape(p_n, r)
                    
                    if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                        return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
            except Exception:
                pass
    
    # Fallback to schoolbook
    U, V, W = _make_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
