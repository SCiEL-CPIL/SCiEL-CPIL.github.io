import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second margin for safety
    
    n, m, p = 2, 3, 4
    # T[a,b,d] where a=i*m+j, b=j*p+k, d=k*n+i
    # T[i*3+j, j*4+k, k*2+i] = 1
    
    # Try ranks from 18 upward
    for rank in range(18, 25):
        if time.time() > deadline - 5:
            break
            
        # Build target tensor
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    d = k * n + i
                    T[a, b, d] = 1.0
        
        def compute_error(U, V, W):
            T_hat = np.einsum('ac,bc,dc->abd', U, V, W)
            return np.linalg.norm(T - T_hat)
        
        def unflatten(x):
            size_u = n * m * rank
            size_v = m * p * rank
            U = x[:size_u].reshape(n * m, rank)
            V = x[size_u:size_u + size_v].reshape(m * p, rank)
            W = x[size_u + size_v:].reshape(p * n, rank)
            return U, V, W
        
        def objective(x):
            U, V, W = unflatten(x)
            return compute_error(U, V, W)
        
        best_solution = None
        best_err = float('inf')
        
        # Multiple random initializations
        np.random.seed(42)
        max_trials = 30
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
                
            x0 = np.random.randn(n * m * rank + m * p * rank + p * n * rank) * 0.1
            
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if res.fun < best_err:
                best_err = res.fun
                best_solution = res.x
                
                if best_err < 1e-10:
                    break
        
        if best_solution is None:
            continue
            
        U, V, W = unflatten(best_solution)
        
        # Round to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify exactness with tight tolerance
        err_r = compute_error(U_r, V_r, W_r)
        
        if err_r < 1e-9:
            # Convert to lists
            U_list = U_r.tolist()
            V_list = V_r.tolist()
            W_list = W_r.tolist()
            
            # Clean up
            for mat in [U_list, V_list, W_list]:
                for i in range(len(mat)):
                    for j in range(len(mat[i])):
                        if abs(mat[i][j]) < 1e-10:
                            mat[i][j] = 0.0
                        else:
                            # Round to nearest 0.5 again for safety
                            mat[i][j] = round(mat[i][j] * 2) / 2.0
                        
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook rank 24
    rank = 24
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
