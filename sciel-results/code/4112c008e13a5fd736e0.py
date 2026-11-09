import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    T_flat = T.flatten()
    
    def compute_error(params, rank):
        U_size = n * m * rank
        V_size = m * p * rank
        W_size = p * n * rank
        
        U = params[:U_size].reshape((n*m, rank))
        V = params[U_size:U_size+V_size].reshape((m*p, rank))
        W = params[U_size+V_size:].reshape((p*n, rank))
        
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        T_rec_flat = T_rec.flatten()
        
        err = T_rec_flat - T_flat
        return err

    def objective(params, rank):
        err = compute_error(params, rank)
        return np.sum(err**2)

    def check_decomposition(U, V, W, tol=1e-9):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(T_rec, T, atol=tol):
            return False
        U_check = 2 * U
        V_check = 2 * V
        W_check = 2 * W
        if not np.allclose(U_check, np.round(U_check), atol=1e-6):
            return False
        if not np.allclose(V_check, np.round(V_check), atol=1e-6):
            return False
        if not np.allclose(W_check, np.round(W_check), atol=1e-6):
            return False
        return True

    def snap_to_half_integer(X):
        return 0.5 * np.round(2 * X)

    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    # Known lower bound for 2x3x4 is 18. Upper bound is 24.
    # We will try to find a solution with rank < 24.
    # If we find one, we return it. Otherwise schoolbook.
    
    best_solution = None
    best_rank = rank_schoolbook

    # We'll iterate through ranks. 
    # To improve search, we can use better initialization.
    # One strategy: Start with schoolbook and try to merge terms?
    # Or random restarts with L-BFGS-B which is more robust for large problems?
    # BFGS can be slow. L-BFGS-B is memory efficient.
    
    # Let's try L-BFGS-B with multiple restarts.
    # Also, we can try to fix the gauge freedom.
    # The tensor decomposition is invariant under U -> U D, V -> V D^{-1}, W -> W.
    # Or U -> U P, V -> V P^T, W -> W.
    # We can fix one column of U to be canonical?
    
    # Let's just do robust random search with L-BFGS-B first.
    
    ranks_to_try = [18, 19, 20, 21, 22, 23]
    
    for rank in ranks_to_try:
        if time.time() - start_time > time_limit:
            break
            
        found = False
        # Number of restarts depends on time
        # Let's try 50 restarts for lower ranks, fewer for higher if time is tight
        num_restarts = 50
        
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            np.random.seed(1000 + rank * 100 + restart)
            
            # Initialization
            # Small random values
            U0 = np.random.randn(n*m, rank) * 0.1
            V0 = np.random.randn(m*p, rank) * 0.1
            W0 = np.random.randn(p*n, rank) * 0.1
            
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                res = minimize(objective, params0, args=(rank,), method='L-BFGS-B', 
                               options={'maxiter': 1000, 'gtol': 1e-8, 'ftol': 1e-12})
            except Exception:
                continue
                
            if res.fun < 1e-6: # Threshold for "close enough" to try snapping
                U_size = n * m * rank
                V_size = m * p * rank
                U = res.x[:U_size].reshape((n*m, rank))
                V = res.x[U_size:U_size+V_size].reshape((m*p, rank))
                W = res.x[U_size+V_size:].reshape((p*n, rank))
                
                U_snap = snap_to_half_integer(U)
                V_snap = snap_to_half_integer(V)
                W_snap = snap_to_half_integer(W)
                
                if check_decomposition(U_snap, V_snap, W_snap):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                else:
                    # If snapping fails, maybe we are close but not quite.
                    # Or maybe the solution is not half-integer (but problem says it must be).
                    # Let's try a local polish around the snapped solution?
                    # Or just accept if error is extremely small?
                    # The problem statement implies exact decomposition with half-integers.
                    # So if snap fails, it's likely a false positive (local min that isn't exact).
                    pass

    # Fallback to schoolbook
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
