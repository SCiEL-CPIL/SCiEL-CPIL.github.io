# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    
    # Target tensor: T[i,j,k] = 1 for all i,j,k
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    def compute_tensor(U_flat, V_flat, W_flat, R):
        U = U_flat.reshape(n, m, R)
        V = V_flat.reshape(m, p, R)
        W = W_flat.reshape(p, n, R)
        
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Use einsum: 'ijr,jkr,kir->ijk'
        T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        return T

    def objective(params, R):
        U_flat = params[:n*m*R]
        V_flat = params[n*m*R : n*m*R + m*p*R]
        W_flat = params[n*m*R + m*p*R :]
        
        T = compute_tensor(U_flat, V_flat, W_flat, R)
        # Target is ones
        error = np.sum((T - 1.0) ** 2)
        return error

    def check_solution(U_flat, V_flat, W_flat, R):
        T = compute_tensor(U_flat, V_flat, W_flat, R)
        if not np.allclose(T, 1.0, atol=1e-6):
            return False
        
        # Try to round to half-integers
        U_r = np.round(U_flat * 2) / 2.0
        V_r = np.round(V_flat * 2) / 2.0
        W_r = np.round(W_flat * 2) / 2.0
        
        T_r = compute_tensor(U_r, V_r, W_r, R)
        if np.allclose(T_r, 1.0, atol=1e-6):
            return True
        return False

    # Try ranks from a lower bound up to schoolbook
    # Lower bound for 3x3x5 is likely around 30.
    # Schoolbook is 45.
    best_rank = n * m * p
    best_sol = None
    
    # We need to be fast. 120s limit.
    # Let's try ranks 30 to 45.
    # For each rank, try a few random starts.
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    for R in range(30, best_rank + 1):
        if time.time() - start_time > time_limit:
            break
            
        # Number of attempts per rank
        max_attempts = 5 if R < 35 else 3
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
                
            np.random.seed(42 + R * 100 + attempt)
            
            # Initialize with random integers or half-integers
            # Range [-1, 1] is good for Strassen-like
            U0 = np.random.uniform(-1, 1, (n*m, R))
            V0 = np.random.uniform(-1, 1, (m*p, R))
            W0 = np.random.uniform(-1, 1, (p*n, R))
            
            # Scale? Usually not needed if we check rounding.
            
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            # Nelder-Mead
            # This can be slow. Let's use a tighter tolerance and maxiter.
            try:
                res = minimize(objective, params0, args=(R,), method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
                
                if res.fun < 1e-6:
                    if check_solution(res.x[:n*m*R], res.x[n*m*R:n*m*R+m*p*R], res.x[n*m*R+m*p*R:], R):
                        U_final = res.x[:n*m*R].reshape(n*m, R)
                        V_final = res.x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                        W_final = res.x[n*m*R+m*p*R:].reshape(p*n, R)
                        
                        # Round again for output
                        U_out = np.round(U_final * 2) / 2.0
                        V_out = np.round(V_final * 2) / 2.0
                        W_out = np.round(W_final * 2) / 2.0
                        
                        return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
            except Exception:
                pass

    # Fallback to schoolbook
    R = best_rank
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
