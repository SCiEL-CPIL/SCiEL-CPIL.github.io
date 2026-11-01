import numpy as np
from scipy.optimize import least_squares
import time

def construct():
    n, m, p = 2, 3, 3
    deadline = time.time() + 140  # safety margin
    
    # Precompute the target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a,b,c] = 1

    # Theoretical lower bound for rank of 2x3x3 matmul is 10.
    # We search from 10 upwards.
    for rank in range(10, 19):
        if time.time() > deadline - 20:
            break
        
        best_sol = None
        best_cost = np.inf
        
        # Significantly reduced trials and max_nfev to avoid timeout
        # The previous 20 trials x 2000 nfev was too slow.
        num_trials = 3 if rank < 12 else 1
        max_nfev = 200
        
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + rank * 100 + trial)
            
            U0 = np.random.randn(n*m, rank)
            V0 = np.random.randn(m*p, rank)
            W0 = np.random.randn(p*n, rank)
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def residual(x):
                U = x[:n*m*rank].reshape(n*m, rank)
                V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                R = np.einsum('as,bs,cs->abc', U, V, W)
                return (R - T).flatten()
            
            try:
                sol = least_squares(residual, x0, max_nfev=max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-8)
                cost = np.sum(sol.fun**2)
                if cost < best_cost:
                    best_cost = cost
                    best_sol = sol.x
                if cost < 1e-8:
                    break
            except Exception:
                pass
        
        if best_sol is not None and best_cost < 1e-8:
            U = best_sol[:n*m*rank].reshape(n*m, rank)
            V = best_sol[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W = best_sol[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            U_out = [[round(x*2)/2 for x in row] for row in U]
            V_out = [[round(x*2)/2 for x in row] for row in V]
            W_out = [[round(x*2)/2 for x in row] for row in W]
            
            U_np = np.array(U_out)
            V_np = np.array(V_out)
            W_np = np.array(W_out)
            R = np.einsum('as,bs,cs->abc', U_np, V_np, W_np)
            if np.allclose(R, T, atol=1e-9):
                return {"U": U_out, "V": V_out, "W": W_out}
    
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
