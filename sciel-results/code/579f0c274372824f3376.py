import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Construct the baseline Rank 24 solution
    # This is always a valid solution.
    rank_base = n * m * p
    U_base = np.zeros((n * m, rank_base), dtype=float)
    V_base = np.zeros((m * p, rank_base), dtype=float)
    W_base = np.zeros((p * n, rank_base), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_base[i * m + j, col] = 1.0
                V_base[j * p + k, col] = 1.0
                W_base[k * n + i, col] = 1.0
                col += 1
                
    best_result = {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
    best_rank = rank_base

    # 2. Define the target tensor
    # T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0

    # 3. Optimization Strategy
    # We try to find a decomposition with rank < 24.
    # We iterate through target ranks from 23 down to 20.
    # For each rank, we perform several random restarts of L-BFGS-B.
    
    def objective_function(params, U_shape, V_shape, W_shape):
        """
        Unpack parameters into U, V, W and compute the squared error
        between the approximated tensor and the target tensor.
        """
        U = params[:U_shape[0]*U_shape[1]].reshape(U_shape)
        V = params[U_shape[0]*U_shape[1]:U_shape[0]*U_shape[1]+V_shape[0]*V_shape[1]].reshape(V_shape)
        W = params[U_shape[0]*U_shape[1]+V_shape[0]*V_shape[1]:].reshape(W_shape)
        
        # Compute tensor product sum: T_approx[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Squared error
        err = np.sum((T_approx - target) ** 2)
        return err

    def try_rank(r, max_restarts=20):
        """
        Attempt to find a valid decomposition for a specific rank r.
        Returns (U, V, W) if successful, else None.
        """
        if time.time() > deadline:
            return None
            
        U_shape = (n * m, r)
        V_shape = (m * p, r)
        W_shape = (p * n, r)
        
        total_vars = U_shape[0]*U_shape[1] + V_shape[0]*V_shape[1] + W_shape[0]*W_shape[1]
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
                
            # Random initialization
            # Scale factors matter. 1.0 is a reasonable start for entries that should be ~1.
            np.random.seed(42 + restart * 101) # Reproducible but varied
            params0 = np.random.randn(total_vars) * 0.5
            
            # Optimization
            # L-BFGS-B is good for medium-sized problems
            res = minimize(
                objective_function, 
                params0, 
                args=(U_shape, V_shape, W_shape),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < 1e-6:
                # Extract solution
                U = res.x[:U_shape[0]*U_shape[1]].reshape(U_shape)
                V = res.x[U_shape[0]*U_shape[1]:U_shape[0]*U_shape[1]+V_shape[0]*V_shape[1]].reshape(V_shape)
                W = res.x[U_shape[0]*U_shape[1]+V_shape[0]*V_shape[1]:].reshape(W_shape)
                
                # Check for near half-integer values
                # The problem allows integer/half-integer entries.
                # We round to nearest 0.5 and check exactness.
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify exactness with rounded values
                T_approx = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                if np.allclose(T_approx, target, atol=1e-9):
                    return U_r, V_r, W_r

        return None

    # Try ranks 23, 22, 21, 20
    # We prioritize lower ranks, but higher ranks might be easier to find?
    # Actually, lower rank is the goal. If rank 20 is found, we stop.
    # However, finding rank 20 is harder than 23.
    # Let's try 23 first as it's closer to the solution manifold?
    # Or just iterate 20->23? 
    # If we find rank 20, we are done.
    # If we don't find 20, we try 21, etc.
    
    for r in [20, 21, 22, 23]:
        if time.time() > deadline - 5:
            break
            
        sol = try_rank(r)
        if sol:
            U, V, W = sol
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = r
            # Since we are iterating from low to high rank, 
            # if we find 20, it's the best possible (assuming lower bound is >= 20).
            # We can break immediately as we found the lowest rank we tried.
            # But maybe 19 is possible? Unlikely for 2x3x4.
            # Let's stick to the tried range.
            break
            
    return best_result
