import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    target_rank = n * m * p
    
    start_time = time.time()
    deadline = start_time + 145.0
    
    def objective(params):
        r = len(params) // 31
        U = params[:n*m*r].reshape(n*m, r)
        V = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
        W = params[n*m*r + m*p*r:].reshape(p*n, r)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                    err += (val - 1.0) ** 2
        return err
    
    def get_matrices(params):
        r = len(params) // 31
        U = params[:n*m*r].reshape(n*m, r)
        V = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
        W = params[n*m*r + m*p*r:].reshape(p*n, r)
        return U, V, W

    def to_list(mat):
        return [[float(val) for val in row] for row in mat]

    # Search from high rank down
    for rank in range(target_rank - 1, 0, -1):
        if time.time() > deadline:
            break
            
        # Determine restarts based on time remaining
        time_left = deadline - time.time()
        # Estimate time per restart ~ 1s? 
        # Let's cap restarts to 20 for first few ranks, less later
        max_restarts = 20 if rank > 25 else 10
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            x0 = np.random.randn(n*m*rank + m*p*rank + p*n*rank) * 0.5
            
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
            
            if res.success and res.fun < 1e-8:
                U, V, W = get_matrices(res.x)
                
                # Check for half-integer solution
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify exactness
                is_exact = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            val = np.dot(U_r[u_idx], V_r[v_idx] * W_r[w_idx])
                            if abs(val - 1.0) > 1e-9:
                                is_exact = False
                                break
                        if not is_exact:
                            break
                    if not is_exact:
                        break
                
                if is_exact:
                    return {"U": to_list(U_r), "V": to_list(V_r), "W": to_list(W_r)}
        
    # Fallback: Schoolbook
    rank = target_rank
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
