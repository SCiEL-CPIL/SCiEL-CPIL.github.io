import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
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
    
    start_time = time.time()
    deadline = start_time + 140
    
    # Try ranks from 9 up to 17 (schoolbook is 18)
    # Theoretical lower bound is max(nm, mp, pn) = 9
    for rank in range(9, 18):
        if time.time() > deadline - 15:
            break
            
        best_sol = None
        best_cost = np.inf
        
        # Multiple random restarts
        max_restarts = 8
        for trial in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(42 + rank * 100 + trial)
            # Scale initialization to avoid numerical issues
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
                sol = least_squares(residual, x0, max_nfev=500, method='trf', ftol=1e-10, xtol=1e-10, gtol=1e-10)
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
            
            # Try to convert to half-integer entries
            U_half = np.round(U * 2) / 2
            V_half = np.round(V * 2) / 2
            W_half = np.round(W * 2) / 2
            
            # Verify the half-integer solution
            R = np.einsum('as,bs,cs->abc', U_half, V_half, W_half)
            if np.allclose(R, T, atol=1e-9):
                U_out = U_half.tolist()
                V_out = V_half.tolist()
                W_out = W_half.tolist()
                return {"U": U_out, "V": V_out, "W": W_out}
            
            # If half-integer doesn't work, try integer rounding
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            
            R = np.einsum('as,bs,cs->abc', U_int, V_int, W_int)
            if np.allclose(R, T, atol=1e-9):
                U_out = U_int.tolist()
                V_out = V_int.tolist()
                W_out = W_int.tolist()
                return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: Schoolbook decomposition of rank n*m*p
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
