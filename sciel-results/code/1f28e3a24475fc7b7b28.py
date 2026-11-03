import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0
    
    def schoolbook(n_, m_, p_):
        """Trivial decomposition for n_ x m_ x p_"""
        rank = n_ * m_ * p_
        U = np.zeros((n_ * m_, rank))
        V = np.zeros((m_ * p_, rank))
        W = np.zeros((p_ * n_, rank))
        col = 0
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    U[i * m_ + j, col] = 1.0
                    V[j * p_ + k, col] = 1.0
                    W[k * n_ + i, col] = 1.0
                    col += 1
        return U, V, W
    
    def verify_decomposition(U, V, W, T_target, tol=1e-8):
        """Check if U, V, W decompose T_target"""
        T_full = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.allclose(T_full, T_target, atol=tol)
    
    def _try_rank_decomposition(T, rank, dim_u, dim_v, dim_w, deadline, n_restarts=20, max_iter=100):
        """
        Try to find a rank decomposition of T with given rank.
        Uses alternating minimization with multiple restarts.
        """
        best_error = np.inf
        best_UVW = None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize randomly
            U = np.random.randn(dim_u, rank)
            V = np.random.randn(dim_v, rank)
            W = np.random.randn(dim_w, rank)
            
            # Normalize
            U /= (np.linalg.norm(U, axis=0, keepdims=True) + 1e-12)
            V /= (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)
            W /= (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
            
            prev_error = np.inf
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                for r in range(rank):
                    T_eff_u = np.einsum('uvw,v,w->u', T, V[:, r], W[:, r])
                    denom = np.dot(V[:, r], V[:, r]) * np.dot(W[:, r], W[:, r])
                    if denom > 1e-12:
                        U[:, r] = T_eff_u / denom
                    
                    T_eff_v = np.einsum('uvw,u,w->v', T, U[:, r], W[:, r])
                    denom = np.dot(U[:, r], U[:, r]) * np.dot(W[:, r], W[:, r])
                    if denom > 1e-12:
                        V[:, r] = T_eff_v / denom
                    
                    T_eff_w = np.einsum('uvw,u,v->w', T, U[:, r], V[:, r])
                    denom = np.dot(U[:, r], U[:, r]) * np.dot(V[:, r], V[:, r])
                    if denom > 1e-12:
                        W[:, r] = T_eff_w / denom
                
                T_approx = np.einsum('ur,vr,wr->uvw', U, V, W)
                error = np.linalg.norm(T_approx - T)
                
                if error < best_error:
                    best_error = error
                    best_UVW = (U.copy(), V.copy(), W.copy())
                
                if error < 1e-10:
                    break
                
                if abs(prev_error - error) < 1e-14:
                    break
                prev_error = error
        
        if best_error < 1e-8:
            return best_UVW
        return None
    
    # Try to find a decomposition with rank < 42
    best_total_rank = n * m * p  # 42
    best_result = None
    
    # Try ranks from 41 down to 25 with fewer restarts for larger ranks
    for rank in range(41, 24, -1):
        if time.time() > deadline - 20:
            break
        
        # Use fewer restarts for larger ranks to save time
        n_restarts = max(5, 30 - (41 - rank))
        found = _try_rank_decomposition(T_target, rank, nm, mp, pn, deadline, n_restarts=n_restarts, max_iter=80)
        if found is not None:
            U_full, V_full, W_full = found
            if verify_decomposition(U_full, V_full, W_full, T_target, tol=1e-8):
                best_total_rank = rank
                best_result = (U_full, V_full, W_full)
                break
    
    if best_result is not None:
        U_full, V_full, W_full = best_result
        # Try to clean up: round to nearest half-integer if close
        U_clean = np.round(U_full * 2) / 2
        V_clean = np.round(V_full * 2) / 2
        W_clean = np.round(W_full * 2) / 2
        
        if verify_decomposition(U_clean, V_clean, W_clean, T_target, tol=1e-8):
            return {"U": U_clean.tolist(), "V": V_clean.tolist(), "W": W_clean.tolist()}
        else:
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback to schoolbook
    U, V, W = schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
