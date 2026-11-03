import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 5]
DEADLINE = 145.0  # 5s margin from 150s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n
    
    # Schoolbook solution as fallback (rank 60)
    rank_sb = n * m * p
    U_sb = np.zeros((N, rank_sb))
    V_sb = np.zeros((M, rank_sb))
    W_sb = np.zeros((P, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    T_target = np.ones((n, m, p))
    
    def reconstruct(U, V, W):
        R = U.shape[1]
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        return T
        
    def check_error(U, V, W):
        T = reconstruct(U, V, W)
        err = np.sum((T - T_target)**2)
        return err, T
    
    def try_rank(R, n_restarts=8):
        from scipy.optimize import minimize
        
        def flatten(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
        def unflatten(x):
            size_U = N * R
            size_V = M * R
            U = x[:size_U].reshape(N, R)
            V = x[size_U:size_U+size_V].reshape(M, R)
            W = x[size_U+size_V:].reshape(P, R)
            return U, V, W
            
        def objective(x):
            U, V, W = unflatten(x)
            err, _ = check_error(U, V, W)
            return err
        
        best_err = np.inf
        best_U = None
        best_V = None
        best_W = None
        
        for restart in range(n_restarts):
            if time.time() > DEADLINE - 15.0:
                break
                
            # Structured initialization: start from schoolbook and project to lower rank
            # Or use random with some structure
            U0 = np.random.randn(N, R) * 0.5
            V0 = np.random.randn(M, R) * 0.5
            W0 = np.random.randn(P, R) * 0.5
            
            # Add a bias toward the schoolbook pattern
            # For each rank term, try to capture one (i,j,k) term
            for r in range(min(R, 30)):
                i = r % n
                j = (r // n) % m
                k = (r // (n*m)) % p
                U0[i*m+j, r] += 1.0
                V0[j*p+k, r] += 1.0
                W0[k*n+i, r] += 1.0
            
            x0 = flatten(U0, V0, W0)
            
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'gtol': 1e-5})
            
            if res.fun < best_err:
                best_err = res.fun
                best_U, best_V, best_W = unflatten(res.x)
                
                if best_err < 1e-10:
                    break
        
        # Try to snap to half-integers
        if best_err < 1e-6 and best_U is not None:
            U_snap = np.round(best_U * 2) / 2.0
            V_snap = np.round(best_V * 2) / 2.0
            W_snap = np.round(best_W * 2) / 2.0
            err_snap, _ = check_error(U_snap, V_snap, W_snap)
            if err_snap < 1e-8:
                return U_snap, V_snap, W_snap, R
            
            # Try integer snap
            U_snap = np.round(best_U)
            V_snap = np.round(best_V)
            W_snap = np.round(best_W)
            err_snap, _ = check_error(U_snap, V_snap, W_snap)
            if err_snap < 1e-8:
                return U_snap, V_snap, W_snap, R
        
        return None

    # Try ranks from 58 down to 45
    for R in range(58, 44, -1):
        if time.time() > DEADLINE - 20.0:
            break
        sol = try_rank(R, n_restarts=6)
        if sol:
            U, V, W, r = sol
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
