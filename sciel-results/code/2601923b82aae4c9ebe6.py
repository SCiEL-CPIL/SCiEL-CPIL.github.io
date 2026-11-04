"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Precompute index triples and their positions
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    u_pos = [i * m + j for i, j, k in indices]
    v_pos = [j * p + k for i, j, k in indices]
    w_pos = [k * n + i for i, j, k in indices]
    
    def verify(U, V, W, tol=1e-8):
        """Verify that the decomposition is exact."""
        max_err = 0.0
        for idx, (i, j, k) in enumerate(indices):
            val = np.dot(U[u_pos[idx], :], V[v_pos[idx], :] * W[w_pos[idx], :])
            max_err = max(max_err, abs(val - 1.0))
        return max_err < tol
    
    def try_rank(rank, max_attempts=50, time_budget=30.0):
        """Try to find a decomposition of given rank."""
        start = time.time()
        for attempt in range(max_attempts):
            if time.time() - start > time_budget:
                break
            if time.time() > deadline - 10:
                break
            
            # Use structured initialization based on known patterns
            np.random.seed(rank * 10000 + attempt * 13 + 7)
            
            # Strategy 1: Sparse random initialization
            if attempt % 3 == 0:
                U = np.zeros((nm, rank))
                V = np.zeros((mp, rank))
                W = np.zeros((pn, rank))
                for r in range(rank):
                    # Each rank-1 component covers some entries
                    u_rows = np.random.choice(nm, size=min(3, nm), replace=False)
                    v_rows = np.random.choice(mp, size=min(4, mp), replace=False)
                    w_rows = np.random.choice(pn, size=min(6, pn), replace=False)
                    for r_ in u_rows:
                        U[r_, r] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
                    for r_ in v_rows:
                        V[r_, r] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
                    for r_ in w_rows:
                        W[r_, r] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
            # Strategy 2: Dense random
            elif attempt % 3 == 1:
                scale = 1.0 / np.sqrt(rank)
                U = np.random.randn(nm, rank) * scale
                V = np.random.randn(mp, rank) * scale
                W = np.random.randn(pn, rank) * scale
            # Strategy 3: Binary-like initialization
            else:
                U = np.random.choice([-1, 0, 1], size=(nm, rank))
                V = np.random.choice([-1, 0, 1], size=(mp, rank))
                W = np.random.choice([-1, 0, 1], size=(pn, rank))
                # Scale to reasonable magnitude
                U = U * np.random.uniform(0.5, 1.0, size=(1, rank))
                V = V * np.random.uniform(0.5, 1.0, size=(1, rank))
                W = W * np.random.uniform(0.5, 1.0, size=(1, rank))
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                # Use LM for local optimization
                result = least_squares(
                    lambda x: residual_func(x, rank),
                    x0,
                    method='lm',
                    max_nfev=500,
                    ftol=1e-14,
                    xtol=1e-14,
                    gtol=1e-14
                )
                
                if result.success and result.cost < 1e-10:
                    U_sol = result.x[:nm * rank].reshape(nm, rank)
                    V_sol = result.x[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
                    W_sol = result.x[nm * rank + mp * rank:].reshape(pn, rank)
                    
                    if verify(U_sol, V_sol, W_sol):
                        return U_sol, V_sol, W_sol
            except Exception:
                continue
        
        return None
    
    def residual_func(params, rank):
        U_flat = params[:nm * rank].reshape(nm, rank)
        V_flat = params[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
        W_flat = params[nm * rank + mp * rank:].reshape(pn, rank)
        
        res = np.empty(len(indices))
        for idx in range(len(indices)):
            res[idx] = np.dot(U_flat[u_pos[idx], :], V_flat[v_pos[idx], :] * W_flat[w_pos[idx], :]) - 1.0
        return res
    
    # Try ranks from the known lower bound upward
    for rank in [19, 20, 21, 22, 23]:
        if time.time() > deadline - 15:
            break
        
        remaining = deadline - 15 - time.time()
        if remaining < 5:
            break
        
        # Allocate time budget based on remaining time
        time_budget = min(remaining - 5, 25.0)
        
        solution = try_rank(rank, max_attempts=100, time_budget=time_budget)
        if solution is not None:
            U_sol, V_sol, W_sol = solution
            
            def to_list(arr):
                lst = []
                for row in arr:
                    row_lst = []
                    for val in row:
                        # Try to round to half-integers
                        rounded = round(val * 2) / 2
                        if abs(val - rounded) < 1e-6:
                            row_lst.append(rounded)
                        else:
                            row_lst.append(float(val))
                    lst.append(row_lst)
                return lst
            
            return {"U": to_list(U_sol), "V": to_list(V_sol), "W": to_list(W_sol)}
    
    # Fallback: naive decomposition with rank = n*m*p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * (m * p) + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
