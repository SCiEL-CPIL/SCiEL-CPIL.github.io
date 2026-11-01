import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute the target tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    T_flat = T.flatten()
    
    start_time = time.time()
    deadline = start_time + 140
    
    # Try ranks from 17 down to 10
    for rank in range(17, 9, -1):
        if time.time() > deadline - 10:
            break
            
        best_sol = None
        best_cost = np.inf
        
        # Number of restarts
        # For rank 17, we can afford more restarts
        max_restarts = 50 if rank == 17 else 20
        
        for trial in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(42 + rank * 100 + trial)
            # Initialize with small random values
            U0 = np.random.randn(nm, rank) * 0.5
            V0 = np.random.randn(mp, rank) * 0.5
            W0 = np.random.randn(pn, rank) * 0.5
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def residual(x):
                U = x[:nm*rank].reshape(nm, rank)
                V = x[nm*rank:nm*rank + mp*rank].reshape(mp, rank)
                W = x[nm*rank + mp*rank:].reshape(pn, rank)
                R = np.einsum('as,bs,cs->abc', U, V, W)
                return (R - T).flatten()
            
            try:
                sol = least_squares(residual, x0, max_nfev=1000, method='trf', ftol=1e-12, xtol=1e-12, gtol=1e-12)
                cost = np.sum(sol.fun**2)
                if cost < best_cost:
                    best_cost = cost
                    best_sol = sol.x
                    if cost < 1e-10:
                        break
            except Exception:
                pass
        
        if best_sol is not None and best_cost < 1e-8:
            U = best_sol[:nm*rank].reshape(nm, rank)
            V = best_sol[nm*rank:nm*rank + mp*rank].reshape(mp, rank)
            W = best_sol[nm*rank + mp*rank:].reshape(pn, rank)
            
            # Snap to half-integers
            U_out = [[round(x*2)/2 for x in row] for row in U]
            V_out = [[round(x*2)/2 for x in row] for row in V]
            W_out = [[round(x*2)/2 for x in row] for row in W]
            
            # Verify
            U_np = np.array(U_out, dtype=float)
            V_np = np.array(V_out, dtype=float)
            W_np = np.array(W_out, dtype=float)
            R = np.einsum('as,bs,cs->abc', U_np, V_np, W_np)
            
            if np.allclose(R, T, atol=1e-9):
                return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: Schoolbook decomposition of rank n*m*p
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
