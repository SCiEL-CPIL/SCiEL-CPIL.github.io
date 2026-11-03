import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [3, 4, 8]

def _tensor_residuals(params, R, n, m, p, lambda_reg=0.1):
    """
    Compute residuals for the tensor decomposition with regularization towards half-integers.
    params: flattened array of U, V, W.
    """
    size_U = n * m * R
    size_V = m * p * R
    size_W = p * n * R
    
    U = params[:size_U].reshape(n * m, R)
    V = params[size_U:size_U + size_V].reshape(m * p, R)
    W = params[size_U + size_V:].reshape(p * n, R)
    
    residuals = []
    
    # 1. Tensor reconstruction error
    # T[i*m+j, k*n+i] should be 1? 
    # Wait, the tensor for matmul is T[i,j,k] = 1.
    # The decomposition is: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # So for each (i,j,k), the sum should be 1.
    # There are n*m*p such equations.
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                val = np.dot(u_row, v_row * w_row)
                residuals.append(1.0 - val)
                idx += 1
    
    # 2. Regularization towards half-integers
    # We want U, V, W entries to be close to k/2.
    # Penalty: (2*x - round(2*x))^2
    # This is non-smooth, so we use a smooth approximation or just simple L2 to nearest 0.5?
    # Actually, least_squares minimizes sum of squares of residuals.
    # We can add residuals for each entry: (2*x - round(2*x))
    # But round is not differentiable.
    # Alternative: Just use the tensor error and hope the solution lands near half-integers.
    # Or, use a softer penalty: (x - 0.5*round(2*x))^2 is 0 at half-integers.
    # Let's add these as additional residuals with a weight.
    
    # For each entry in U, V, W
    for mat in [U, V, W]:
        for val in mat.flatten():
            # Distance to nearest half-integer
            # nearest = 0.5 * np.round(val * 2)
            # dist = val - nearest
            # residuals.append(lambda_reg * dist)
            # This adds 3 * (n*m*R + m*p*R + p*n*R) residuals.
            # For R=32, n=3, m=4, p=8:
            # U: 12*32=384, V: 32*32=1024, W: 24*32=768. Total ~2176 residuals.
            # This might slow down the optimizer but guide it.
            pass # Skipping complex regularization for now to keep speed high, relying on post-snap verification.
            
    return np.array(residuals)

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if U @ V @ W^T reconstructs the matmul tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_half_integer(M):
    """Snap matrix entries to nearest 0.5."""
    return np.round(M * 2.0) / 2.0

def _try_rank(R, n, m, p, time_limit, seed_base=0):
    """
    Try to find a decomposition of rank R.
    """
    start_time = time.time()
    max_attempts = 20 # Reduce attempts to allow more ranks to be tried?
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
            
        seed = seed_base + attempt * 137
        np.random.seed(seed)
        
        # Initialize
        # Scale initialization: values around 1/sqrt(R) or similar?
        # If rank is R, and we want sum of products to be 1,
        # and entries are ~1, then sum of R terms is ~R.
        # So we should scale by 1/sqrt(R) or 1/R?
        # If U, V, W are ~ 1/sqrt(R), product is ~ 1/R, sum is ~ 1.
        # Let's try 1/sqrt(R).
        scale = 1.0 / np.sqrt(R)
        
        U_init = np.random.randn(n * m, R) * scale
        V_init = np.random.randn(m * p, R) * scale
        W_init = np.random.randn(p * n, R) * scale
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Run optimization
        try:
            result = opt.least_squares(
                _tensor_residuals, 
                params_init, 
                args=(R, n, m, p),
                method='trf', 
                max_nfev=500 # Reduced to allow more restarts
            )
        except Exception:
            continue
            
        if result.cost < 1e-6:
            # Extract solution
            params_sol = result.x
            size_U = n * m * R
            size_V = m * p * R
            size_W = p * n * R
            
            U_sol = params_sol[:size_U].reshape(n * m, R)
            V_sol = params_sol[size_U:size_U + size_V].reshape(m * p, R)
            W_sol = params_sol[size_U + size_V:].reshape(p * n, R)
            
            # Check if it's a valid solution (real numbers)
            if _verify(U_sol, V_sol, W_sol, n, m, p):
                # Snap to half-integers
                U_snap = _snap_to_half_integer(U_sol)
                V_snap = _snap_to_half_integer(V_sol)
                W_snap = _snap_to_half_integer(W_sol)
                
                if _verify(U_snap, V_snap, W_snap, n, m, p):
                    return U_snap, V_snap, W_snap
                else:
                    # If snap fails, try a local search around the snapped solution?
                    # Or just accept the failure for this attempt.
                    pass
                    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145 # 5s safety margin
    
    # Schoolbook fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Search for lower rank
    # Lower bound: m * max(n, p) is 32? 
    # Actually, rank >= m * min(n, p) is 12? No.
    # Let's start from m * min(n, p) = 12? 
    # But 12 is likely too low.
    # Let's start from m * max(n, p) // 2? 
    # Let's just start from a reasonable guess.
    # For 3x4x8, rank is likely around 30.
    # Let's try from 20 to 96.
    
    min_rank = 20
    max_rank = schoolbook_rank
    
    for R in range(min_rank, max_rank + 1):
        if time.time() > deadline:
            break
            
        time_left = deadline - time.time()
        if time_left < 2:
            break
            
        # Allocate time based on remaining ranks
        # Give more time to lower ranks?
        # Or fixed time per rank?
        # Let's give 1 second per rank.
        
        result = _try_rank(R, n, m, p, time_limit=min(1.0, time_left), seed_base=R)
        
        if result is not None:
            U_sol, V_sol, W_sol = result
            best_result = {
                "U": U_sol.tolist(),
                "V": V_sol.tolist(),
                "W": W_sol.tolist()
            }
            return best_result
            
    return best_result
