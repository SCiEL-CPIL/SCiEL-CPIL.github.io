import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 3, 3
    
    # Build target tensor T
    # Dimensions: (n*m, m*p, p*n) -> (6, 9, 6)
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    T = np.zeros((dim_u, dim_v, dim_w))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def verify(U, V, W):
        """
        U: (dim_u, rank)
        V: (dim_v, rank)
        W: (dim_w, rank)
        """
        rank = U.shape[1]
        # Compute reconstruction
        # Rec[u, v, w] = sum_c U[u, c] V[v, c] W[w, c]
        # This is a tensor product sum.
        # We can compute this efficiently using einsum or loops.
        # For small dims, loops are fine.
        rec = np.zeros((dim_u, dim_v, dim_w))
        for c in range(rank):
            rec += np.einsum('u,v,w->uvw', U[:, c], V[:, c], W[:, c])
        
        # Check exact match
        if not np.allclose(rec, T, atol=1e-9):
            return False
        return True

    def schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, col_idx] = 1.0
                    V[v, col_idx] = 1.0
                    W[w, col_idx] = 1.0
                    col_idx += 1
        return {"U": U, "V": V, "W": W}

    best_result = schoolbook()
    best_rank = n * m * p

    # Try to find lower rank
    for target_rank in [17, 16]:
        if time.time() > deadline:
            break
            
        found = False
        # Number of random restarts
        num_restarts = 5
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize
            U = np.random.rand(dim_u, target_rank)
            V = np.random.rand(dim_v, target_rank)
            W = np.random.rand(dim_w, target_rank)
            
            # Scale initialization
            U *= 1.0
            V *= 1.0
            W *= 1.0
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                if time.time() > deadline:
                    return 1e10
                U_x = x[:dim_u * target_rank].reshape(dim_u, target_rank)
                V_x = x[dim_u * target_rank : dim_u * target_rank + dim_v * target_rank].reshape(dim_v, target_rank)
                W_x = x[dim_u * target_rank + dim_v * target_rank :].reshape(dim_w, target_rank)
                
                rec = np.zeros((dim_u, dim_v, dim_w))
                for c in range(target_rank):
                    rec += np.einsum('u,v,w->uvw', U_x[:, c], V_x[:, c], W_x[:, c])
                
                error = np.sum((rec - T)**2)
                return error
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < 1e-9:
                    U_sol = res.x[:dim_u * target_rank].reshape(dim_u, target_rank)
                    V_sol = res.x[dim_u * target_rank : dim_u * target_rank + dim_v * target_rank].reshape(dim_v, target_rank)
                    W_sol = res.x[dim_u * target_rank + dim_v * target_rank :].reshape(dim_w, target_rank)
                    
                    # Try to round to half-integers
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    if verify(U_rounded, V_rounded, W_rounded):
                        best_result = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                        best_rank = target_rank
                        found = True
                        break
                    elif verify(U_sol, V_sol, W_sol):
                        # If floating point solution is exact enough, return it? 
                        # Task says "integer/half-integer entries". 
                        # So we MUST return rounded if possible.
                        # If not rounded, it might be invalid.
                        pass
                        
            except Exception:
                continue
        
        if found:
            break

    if best_rank < n * m * p:
        return best_result
    else:
        return schoolbook()
