import numpy as np
import time
import random

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 18
    # Fallback ranks
    fallback_ranks = [19, 20]
    
    start_time = time.time()
    time_limit = 110  # Safety margin for 120s limit
    
    best_result = None
    best_score = float('inf')
    
    # Schoolbook solution as ultimate fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
        
    schoolbook = get_schoolbook()
    
    # Define the objective function (Vectorized)
    def objective(params, rank):
        nm = n * m
        mp = m * p
        pn = p * n
        U = params[:rank * nm].reshape(rank, nm).T # (nm, rank)
        V = params[rank * nm : rank * (nm + mp)].reshape(rank, mp).T
        W = params[rank * (nm + mp) :].reshape(rank, pn).T
        
        # Reshape for einsum
        # U: (n, m, rank)
        # V: (m, p, rank)
        # W: (p, n, rank)
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        # T_ijk = sum_r U_ijr V_jkr W_kir
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        
        error = np.sum((T - 1.0) ** 2)
        return error

    def try_rank(rank):
        if time.time() - start_time > time_limit:
            return None
            
        nm = n * m
        mp = m * p
        pn = p * n
        total_params = rank * (nm + mp + pn)
        
        # Initialize with random small values
        # Use random restarts
        max_restarts = 20 # Reduced to allow more time per restart or more ranks
        
        for r in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Random initialization
            params = np.random.uniform(-1, 1, total_params)
            
            # Simple local search using Nelder-Mead
            # scipy.optimize is available
            from scipy.optimize import minimize
            
            try:
                res = minimize(objective, params, args=(rank,), method='Nelder-Mead', 
                               options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 500, 'maxfev': 1000})
                
                if res.fun < 1e-6:
                    # Found a solution
                    # Extract U, V, W
                    U = res.x[:rank * nm].reshape(rank, nm).T
                    V = res.x[rank * nm : rank * (nm + mp)].reshape(rank, mp).T
                    W = res.x[rank * (nm + mp) :].reshape(rank, pn).T
                    
                    # Try to snap to half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    # Verify snap
                    # Use vectorized check for speed
                    U_r = U_snap.reshape(n, m, rank)
                    V_r = V_snap.reshape(m, p, rank)
                    W_r = W_snap.reshape(p, n, rank)
                    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                    err_snap = np.sum((T - 1.0) ** 2)
                    
                    if err_snap < 1e-6:
                        # Valid half-integer solution
                        # Return NumPy arrays to ensure shape/type consistency
                        return {"U": U_snap, "V": V_snap, "W": W_snap}
                    else:
                        # If snap fails, maybe return floats? 
                        # The task says "integer/half-integer entries". 
                        # If I return floats, it might be invalid. 
                        # Let's stick to snapped if possible. 
                        pass
            except Exception as e:
                pass
                
        return None

    # Try rank 18
    result = try_rank(18)
    if result:
        return result
        
    # Try rank 19
    if time.time() - start_time < time_limit:
        result = try_rank(19)
        if result:
            return result
            
    # Fallback to schoolbook
    return schoolbook
