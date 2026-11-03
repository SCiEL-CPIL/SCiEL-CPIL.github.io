import numpy as np
import time
import itertools
import sys

DIMENSIONS = [3, 4, 5]
DEADLINE = 145.0 # 5s margin

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n
    
    # Schoolbook solution as fallback
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
    
    # Precompute target tensor
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
        
    def try_rank(R):
        # Random initialization
        best_err = np.inf
        best_U = None
        best_V = None
        best_W = None
        
        # Number of restarts based on remaining time
        # Each optimization step might take ~0.1s to 1s
        # Let's try a few simple gradient steps or just random search with local polish
        # For small R, random search with local refinement can work.
        
        # Since we need EXACT integer/half-integer, we can use a grid search on a small lattice?
        # No, R is up to 49. Dimension of U is 12*49 = 588. Too large for grid.
        
        # Use scipy.optimize?
        # We can flatten U, V, W into a single vector x of length R*(N+M+P)
        
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
            
        # More restarts with better initialization
        n_restarts = 10
        for restart in range(n_restarts):
            if time.time() > DEADLINE - 15.0:
                break
                
            # Random init with scale 1.0
            U0 = np.random.randn(N, R)
            V0 = np.random.randn(M, R)
            W0 = np.random.randn(P, R)
            
            x0 = flatten(U0, V0, W0)
            
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 300, 'gtol': 1e-6})
            
            if res.fun < best_err:
                best_err = res.fun
                best_U, best_V, best_W = unflatten(res.x)
                
                # Early termination if error is tiny
                if best_err < 1e-10:
                    break
                
        # If we found a very small error, try to snap to rationals
        if best_err < 1e-8:
            U_int, V_int, W_int = best_U, best_V, best_W
            # Try to snap to half-integers (multiply by 2, round, divide by 2)
            # Or integers
            # Check if snapping works
            U_snap = np.round(U_int * 2) / 2.0
            V_snap = np.round(V_int * 2) / 2.0
            W_snap = np.round(W_int * 2) / 2.0
            err_snap, _ = check_error(U_snap, V_snap, W_snap)
            if err_snap < 1e-8:
                return U_snap, V_snap, W_snap, R
            else:
                # Try integer snap
                U_snap = np.round(U_int)
                V_snap = np.round(V_int)
                W_snap = np.round(W_int)
                err_snap, _ = check_error(U_snap, V_snap, W_snap)
                if err_snap < 1e-8:
                    return U_snap, V_snap, W_snap, R
        return None

    # Try ranks from 55 down to 40
    for R in range(55, 39, -1):
        if time.time() > DEADLINE - 25.0:
            break
        sol = try_rank(R)
        if sol:
            U, V, W, r = sol
            # Convert to list of lists as required
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
