import numpy as np
from scipy.optimize import minimize
import time

def construct():
    n, m, p = 2, 4, 5
    start_time = time.time()
    
    # Target tensor: T[i,j,k] = 1 for all i,j,k
    # U: (n*m, r) = (8, r)
    # V: (m*p, r) = (20, r)
    # W: (p*n, r) = (10, r)
    # Constraint: sum_r U[4i+j, r] * V[5j+k, r] * W[2k+i, r] = 1
    
    # Try ranks from 32 to 40
    for r in range(32, 41):
        if time.time() - start_time > 100:
            break
        # Initialize with schoolbook (first r terms)
        U = np.zeros((8, r))
        V = np.zeros((20, r))
        W = np.zeros((10, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < r:
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
        
        # Define residual
        def residual(params):
            U_flat = params[:8*r].reshape(8, r)
            V_flat = params[8*r:8*r+20*r].reshape(20, r)
            W_flat = params[8*r+20*r:].reshape(10, r)
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_flat[i*m+j, :] * V_flat[j*p+k, :] * W_flat[k*n+i, :])
                        res.append(val - 1)
            return np.array(res)
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use least squares
        from scipy.optimize import least_squares
        result = least_squares(residual, params, max_nfev=1000)
        
        U_opt = result.x[:8*r].reshape(8, r)
        V_opt = result.x[8*r:8*r+20*r].reshape(20, r)
        W_opt = result.x[8*r+20*r:].reshape(10, r)
        
        # Check if close to integer/half-integer
        # Round to nearest 0.5
        U_round = np.round(U_opt * 2) / 2
        V_round = np.round(V_opt * 2) / 2
        W_round = np.round(W_opt * 2) / 2
        
        # Verify
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_round[i*m+j, :] * V_round[j*p+k, :] * W_round[k*n+i, :])
                    max_err = max(max_err, abs(val - 1))
        
        if max_err < 1e-6:
            # Convert to list of lists
            U_list = [[float(x) for x in row] for row in U_round]
            V_list = [[float(x) for x in row] for row in V_round]
            W_list = [[float(x) for x in row] for row in W_round]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
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
