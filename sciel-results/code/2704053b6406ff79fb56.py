"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24
    
    deadline = time.time() + 145  # safety margin before 150s kill
    
    best_rank = n * m * p  # start with schoolbook (120)
    best_result = None
    
    # First, provide the schoolbook as fallback
    schoolbook_U = np.zeros((nm, best_rank))
    schoolbook_V = np.zeros((mp, best_rank))
    schoolbook_W = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                schoolbook_U[i * m + j, col] = 1
                schoolbook_V[j * p + k, col] = 1
                schoolbook_W[k * n + i, col] = 1
                col += 1
    best_result = {
        "U": schoolbook_U.tolist(),
        "V": schoolbook_V.tolist(),
        "W": schoolbook_W.tolist()
    }
    
    np.random.seed(42)
    
    def check_error(U, V, W, tol=1e-9):
        """Check max error of tensor decomposition using vectorized einsum."""
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        max_err = np.max(np.abs(T - 1.0))
        return float(max_err)
    
    def try_half_integer(U, V, W, max_trials=50):
        """Try to find a half-integer solution near the given solution."""
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        err = check_error(U_round, V_round, W_round)
        if err < 1e-9:
            return U_round, V_round, W_round
        
        for _ in range(max_trials):
            noise_u = np.random.randn(nm, U.shape[1]) * 0.05
            noise_v = np.random.randn(mp, V.shape[1]) * 0.05
            noise_w = np.random.randn(pn, W.shape[1]) * 0.05
            U_t = np.round((U + noise_u) * 2) / 2
            V_t = np.round((V + noise_v) * 2) / 2
            W_t = np.round((W + noise_w) * 2) / 2
            err = check_error(U_t, V_t, W_t)
            if err < 1e-9:
                return U_t, V_t, W_t
        return None, None, None
    
    # Vectorized ALS Update
    def update_U(U, V, W, rank):
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # A[i,j,k,r] = V[j,k,r] * W[k,i,r]
        # Shape: (n, m, p, rank)
        A = np.einsum('jkr,kir->ijkr', V_3d, W_3d)
        
        # For each i,j, solve A[i,j] @ x = ones(p)
        # A[i,j] is (p, rank)
        for i in range(n):
            for j in range(m):
                try:
                    x = np.linalg.lstsq(A[i, j], np.ones(p), rcond=None)[0]
                    U_3d[i, j, :] = x
                except:
                    pass
        return U_3d.reshape(nm, rank)

    def update_V(U, V, W, rank):
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # A[j,k,i,r] = U[i,j,r] * W[k,i,r]
        # Shape: (m, p, n, rank)
        A = np.einsum('ijr,kir->jkir', U_3d, W_3d)
        
        for j in range(m):
            for k in range(p):
                try:
                    x = np.linalg.lstsq(A[j, k], np.ones(n), rcond=None)[0]
                    V_3d[j, k, :] = x
                except:
                    pass
        return V_3d.reshape(mp, rank)

    def update_W(U, V, W, rank):
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # A[k,i,j,r] = U[i,j,r] * V[j,k,r]
        # Shape: (p, n, m, rank)
        A = np.einsum('ijr,jkr->kijr', U_3d, V_3d)
        
        for k in range(p):
            for i in range(n):
                try:
                    x = np.linalg.lstsq(A[k, i], np.ones(m), rcond=None)[0]
                    W_3d[k, i, :] = x
                except:
                    pass
        return W_3d.reshape(pn, rank)

    # Search ranks from 110 to 119
    # The inspiration suggests 112 is possible.
    for rank in range(110, 120):
        if time.time() > deadline - 20:
            break
        
        restarts = 20 if rank < 115 else 10
        
        for restart in range(restarts):
            if time.time() > deadline - 10:
                break
            
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            converged = False
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                
                U = update_U(U, V, W, rank)
                V = update_V(U, V, W, rank)
                W = update_W(U, V, W, rank)
                
                max_err = check_error(U, V, W)
                if max_err < 1e-9:
                    converged = True
                    break
            
            if converged:
                U_h, V_h, W_h = try_half_integer(U, V, W, max_trials=50)
                if U_h is not None:
                    best_rank = rank
                    best_result = {
                        "U": U_h.tolist(),
                        "V": V_h.tolist(),
                        "W": W_h.tolist()
                    }
                    break
        
        if best_rank < n * m * p:
            break
    
    return best_result
# EVOLVE-BLOCK-END
