import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Precompute indices
    i_idx = np.array([i for i in range(n) for j in range(m) for k in range(p)])
    j_idx = np.array([j for i in range(n) for j in range(m) for k in range(p)])
    k_idx = np.array([k for i in range(n) for j in range(m) for k in range(p)])
    
    u_rows = i_idx * m + j_idx  # shape (n*m*p,)
    v_rows = j_idx * p + k_idx  # shape (n*m*p,)
    w_rows = k_idx * n + i_idx  # shape (n*m*p,)
    
    for rank in range(18, 55):
        if time.time() > deadline - 15:
            break
        
        best_error = float('inf')
        best_params = None
        
        max_attempts = 2
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(42 + attempt * 7)
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                Ux = x[:n*m*rank].reshape(n*m, rank)
                Vx = x[n*m*rank:(n*m+m*p)*rank].reshape(m*p, rank)
                Wx = x[(n*m+m*p)*rank:].reshape(p*n, rank)
                
                terms = np.zeros(n*m*p)
                for r in range(rank):
                    terms += Ux[u_rows, r] * Vx[v_rows, r] * Wx[w_rows, r]
                
                error = np.sum((terms - 1.0) ** 2)
                return error
            
            result = minimize(objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14})
            
            if result.fun < best_error:
                best_error = result.fun
                best_params = result.x
        
        if best_params is not None and best_error < 1e-6:
            U = best_params[:n*m*rank].reshape(n*m, rank)
            V = best_params[n*m*rank:(n*m+m*p)*rank].reshape(m*p, rank)
            W = best_params[(n*m+m*p)*rank:].reshape(p*n, rank)
            
            # Round to nearest half-integer
            U = 2 * np.round(U / 2)
            V = 2 * np.round(V / 2)
            W = 2 * np.round(W / 2)
            
            # Verify exactness with rounded values
            terms = np.zeros(n*m*p)
            for r in range(rank):
                terms += U[u_rows, r] * V[v_rows, r] * W[w_rows, r]
            
            if np.allclose(terms, 1.0, atol=1e-9):
                U_list = [[float(U[i,j]) for j in range(rank)] for i in range(n*m)]
                V_list = [[float(V[i,j]) for j in range(rank)] for i in range(m*p)]
                W_list = [[float(W[i,j]) for j in range(rank)] for i in range(p*n)]
                
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
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
