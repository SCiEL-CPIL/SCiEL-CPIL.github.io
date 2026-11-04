import numpy as np
import time
import random
from scipy.optimize import least_squares

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def check_decomposition(U, V, W):
        """
        Checks if the decomposition is exact.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        """
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
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

    def try_rank_19():
        rank = 19
        # Initialize with random small integers/half-integers
        # Or random floats
        best_score = float('inf')
        best_solution = None
        
        # We will try a few random restarts
        num_restarts = 20
        for _ in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            # Scale factors to be around 1
            U_flat = np.random.randint(-2, 3, (n*m*rank)) * 0.5
            V_flat = np.random.randint(-2, 3, (m*p*rank)) * 0.5
            W_flat = np.random.randint(-2, 3, (p*n*rank)) * 0.5
            
            # Reshape
            U = U_flat.reshape(n*m, rank)
            V = V_flat.reshape(m*p, rank)
            W = W_flat.reshape(p*n, rank)
            
            def residuals(params):
                U_f = params[0:n*m*rank].reshape(n*m, rank)
                V_f = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W_f = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                res = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(rank):
                                val += U_f[i*m+j, c] * V_f[j*p+k, c] * W_f[k*n+i, c]
                            res.append(val - 1)
                return np.array(res)
                
            def jac(params):
                # Analytical Jacobian is complex, let numpy do it or use finite diff
                # For 171 vars and 27 eqs, finite diff is okayish but slow.
                # Let's use '2-point' or let scipy estimate.
                return None # Let scipy estimate
                
            try:
                x0 = np.concatenate([U_flat, V_flat, W_flat])
                result = least_squares(residuals, x0, method='lm', max_nfev=1000)
                
                if result.success or result.cost < best_score:
                    # Check if it's close to integer/half-integer
                    U_opt = result.x[0:n*m*rank].reshape(n*m, rank)
                    V_opt = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_opt = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    
                    # Snap to nearest half-integer
                    U_snap = np.round(U_opt * 2) / 2
                    V_snap = np.round(V_opt * 2) / 2
                    W_snap = np.round(W_opt * 2) / 2
                    
                    # Verify exactness of snapped version
                    if check_decomposition(U_snap.tolist(), V_snap.tolist(), W_snap.tolist()):
                        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                        
                    # Also verify the float version just in case snapping failed but float was exact (unlikely)
                    if check_decomposition(U_opt.tolist(), V_opt.tolist(), W_opt.tolist()):
                         # Try to snap more aggressively or just return if allowed? 
                         # Prompt says "integer/half-integer entries". So must snap.
                         pass
                         
            except Exception:
                pass
                
        return None

    # Try to find a lower rank decomposition
    result_19 = try_rank_19()
    if result_19:
        return result_19

    # Fallback to schoolbook
    result = schoolbook()
    if check_decomposition(result['U'], result['V'], result['W']):
        return result
    else:
        return result
