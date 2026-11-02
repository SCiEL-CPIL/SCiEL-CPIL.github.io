import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    start_time = time.time()
    deadline = start_time + 100.0 # Leave 20s margin
    
    # Try to find a lower rank decomposition
    # We iterate over possible ranks
    # Lower bound is max(n*m, m*p, p*n) = 25
    # We try a few ranks below 75
    
    for rank in range(30, 75):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W with random values
            # We use small random values to avoid divergence
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Define the objective function
            def objective(params):
                # params is a flat array of length n*m*rank + m*p*rank + p*n*rank
                size_U = n * m * rank
                size_V = m * p * rank
                size_W = p * n * rank
                
                U_flat = params[:size_U]
                V_flat = params[size_U:size_U+size_V]
                W_flat = params[size_U+size_V:]
                
                U_mat = U_flat.reshape(n * m, rank)
                V_mat = V_flat.reshape(m * p, rank)
                W_mat = W_flat.reshape(p * n, rank)
                
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(rank):
                                val += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                            diff = 1.0 - val
                            error += diff * diff
                return error
            
            # We don't have scipy minimize easily accessible in all environments without import, 
            # but the prompt says NumPy and SciPy are available.
            try:
                from scipy.optimize import minimize
            except ImportError:
                # Fallback: simple gradient descent or just skip optimization
                continue
                
            params_init = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use a simple optimizer. L-BFGS-B is good.
            # But it might be slow. Let's try a few steps.
            # Actually, for 120s, we can afford a few full optimizations.
            
            # To speed up, we can check if the initial random guess is already good (unlikely).
            # Let's just run minimize.
            
            res = minimize(objective, params_init, method='L-BFGS-B', options={'maxiter': 50, 'gtol': 1e-8})
            
            if res.success and res.fun < 1e-6:
                # Check if we can snap to integers/half-integers
                U_opt = res.x[:n*m*rank].reshape(n*m, rank)
                V_opt = res.x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                W_opt = res.x[n*m*rank + m*p*rank :].reshape(p*n, rank)
                
                # Snap to nearest 0.5
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                # Verify exact decomposition with snapped values
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(rank):
                                val += U_snap[i*m+j, r] * V_snap[j*p+k, r] * W_snap[k*n+i, r]
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                        
                if valid:
                    # Convert to list of lists
                    U_list = U_snap.tolist()
                    V_list = V_snap.tolist()
                    W_list = W_snap.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
    return best_result
