import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    # Precompute indices for the tensor equations
    # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # for all i in [0,n), j in [0,m), k in [0,p)
    
    # Define objective function
    def objective(params):
        if time.time() > deadline:
            return 1e10
        U = params[:n*m*R].reshape(n*m, R)
        V = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W = params[n*m*R + m*p*R:].reshape(p*n, R)
        
        # Calculate error
        # To speed up, we can use einsum or manual loops
        # T_ijk = sum_r U_im+j, r * V_jp+k, r * W_kn+i, r
        # We want T_ijk == 1
        
        # Using einsum for speed
        # U is (n*m, R), V is (m*p, R), W is (p*n, R)
        # We need to align indices.
        # This is tricky with einsum directly because of the index permutations.
        
        # Let's use a loop for clarity and moderate speed, or vectorized ops.
        # n=2, m=3, p=8. 48 equations.
        # R is small (40-47).
        # 48 * R operations per evaluation. Very fast.
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.dot(U[u_idx], V[v_idx] * W[w_idx]) # No, sum over r of U_r * V_r * W_r
                    # s = sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    s = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                    err += (s - 1.0) ** 2
        return err

    # We need to try different ranks
    best_rank = n * m * p
    best_solution = None
    
    # Try ranks from 47 down to 40
    for R in range(47, 39, -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few random starts
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(attempt + R)
            # Initialize with small random values
            params0 = np.random.randn(n*m*R + m*p*R + p*n*R) * 0.1
            
            # Add schoolbook structure as a hint?
            # Schoolbook has rank 48.
            # We can project schoolbook to rank R?
            # Or just use random.
            
            res = minimize(objective, params0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
            
            if res.fun < 1e-6:
                # Check if it can be snapped to half-integers
                U = res.x[:n*m*R].reshape(n*m, R)
                V = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W = res.x[n*m*R + m*p*R:].reshape(p*n, R)
                
                # Snap to nearest half-integer
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify snapped solution
                err_snap = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            s = np.sum(U_snap[u_idx] * V_snap[v_idx] * W_snap[w_idx])
                            err_snap += (s - 1.0) ** 2
                
                if err_snap < 1e-6:
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                else:
                    # If snapped fails, maybe the real solution is good enough?
                    # The task requires "integer/half-integer entries".
                    # So we must return snapped if it works.
                    # If it doesn't work, we can't return it.
                    pass

    # If no better solution found, return schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
