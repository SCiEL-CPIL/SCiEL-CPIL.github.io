import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0

    def compute_error(U, V, W):
        # T_approx = sum_r U[:,r] V[:,r] W[:,r] (outer product sum)
        # Using einsum for efficiency
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T_approx - target)**2)

    def check_decomposition(U, V, W, atol=1e-9):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, target, atol=atol)

    def try_rank(r, num_restarts=50, time_limit=100):
        start_time = time.time()
        best_err = float('inf')
        best_solution = None
        
        for restart in range(num_restarts):
            if time.time() > start_time + time_limit:
                break
                
            # Initialization
            # Sparse random initialization often works better for tensor decompositions
            scale = 0.5
            U = np.random.randn(nm, r) * scale
            V = np.random.randn(mp, r) * scale
            W = np.random.randn(pn, r) * scale
            
            # Flatten parameters
            def unpack(params):
                U = params[:nm*r].reshape(nm, r)
                V = params[nm*r:nm*r+mp*r].reshape(mp, r)
                W = params[nm*r+mp*r:].reshape(pn, r)
                return U, V, W

            def objective(params):
                U, V, W = unpack(params)
                return compute_error(U, V, W)

            def gradient(params):
                U, V, W = unpack(params)
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T_approx - target
                
                dU = np.einsum('abc,br,cr->ar', diff, V, W)
                dV = np.einsum('abc,ar,cr->br', diff, U, W)
                dW = np.einsum('abc,ar,br->cr', diff, U, V)
                
                return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])

            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(objective, params0, jac=gradient, method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if res.fun < best_err:
                    best_err = res.fun
                    best_solution = res.x.copy()
                    
                    U, V, W = unpack(res.x)
                    
                    # If error is very low, try to round to half-integers
                    if best_err < 1e-6:
                        # Round to nearest 0.5
                        U_rounded = np.round(U * 2) / 2
                        V_rounded = np.round(V * 2) / 2
                        W_rounded = np.round(W * 2) / 2
                        
                        if check_decomposition(U_rounded, V_rounded, W_rounded, atol=1e-9):
                            return U_rounded, V_rounded, W_rounded
                        
                        # Try rounding to integers
                        U_int = np.round(U)
                        V_int = np.round(V)
                        W_int = np.round(W)
                        if check_decomposition(U_int, V_int, W_int, atol=1e-9):
                            return U_int, V_int, W_int

                        # Try local linear solve around the rounded solution?
                        # This is complex, so we just mark it as a candidate for further polishing if needed
                        # For now, if we get close, we might try a few more restarts or accept if no better found
            except Exception:
                pass
        
        # If we found a very low error but not exact, we might try to force exactness
        # But for now, return None if not exact
        if best_solution is not None and best_err < 1e-6:
            U, V, W = unpack(best_solution)
            # One last attempt to snap
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            if check_decomposition(U_snap, V_snap, W_snap, atol=1e-9):
                return U_snap, V_snap, W_snap
                
        return None

    # Try ranks from low to high
    for rank in [14, 15, 16, 17, 18]:
        sol = try_rank(rank)
        if sol is not None:
            U, V, W = sol
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback: Schoolbook
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
