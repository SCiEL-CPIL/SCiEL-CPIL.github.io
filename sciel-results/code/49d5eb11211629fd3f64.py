import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    deadline = time.time() + 110
    n, m, p = 3, 4, 7
    
    # Dimensions of U, V, W
    dim_U = n * m  # 12
    dim_V = m * p  # 28
    dim_W = p * n  # 21
    
    def verify(U, V, W, tol=1e-5):
        r = U.shape[1]
        # Check equations: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k, r] == delta(j, j')
        # Note: V index is j'*p+k. In the product (sum_j U_ij A_ij) (sum_j' V_j'k B_j'k),
        # the coefficient of A_ij B_j'k is sum_r W_ik U_ij V_j'k.
        # We need this to be 1 if j==j' and 0 otherwise.
        
        errors = 0.0
        max_err = 0.0
        for i in range(n):
            for k in range(p):
                w_idx = k * n + i
                for j in range(m):
                    u_idx = i * m + j
                    for j_prime in range(m):
                        v_idx = j_prime * p + k
                        val = 0.0
                        for r in range(r):
                            val += W[w_idx, r] * U[u_idx, r] * V[v_idx, r]
                        target = 1.0 if j == j_prime else 0.0
                        err = val - target
                        max_err = max(max_err, abs(err))
        return max_err < tol

    def residual(params, r):
        # params is a flat array of length r * (dim_U + dim_V + dim_W)
        U = params[0:r*dim_U].reshape(r, dim_U).T  # (dim_U, r)
        V = params[r*dim_U:r*(dim_U+dim_V)].reshape(r, dim_V).T  # (dim_V, r)
        W = params[r*(dim_U+dim_V):].reshape(r, dim_W).T  # (dim_W, r)
        
        res = []
        for i in range(n):
            for k in range(p):
                w_idx = k * n + i
                for j in range(m):
                    u_idx = i * m + j
                    for j_prime in range(m):
                        v_idx = j_prime * p + k
                        val = np.sum(W[w_idx, :] * U[u_idx, :] * V[v_idx, :])
                        target = 1.0 if j == j_prime else 0.0
                        res.append(val - target)
        return np.array(res)

    # Try ranks from 80 down to 30
    for r in range(80, 29, -1):
        if time.time() > deadline - 10:
            break
            
        # Multiple random restarts
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(42 + restart)
            # Initialize with small random values
            U_init = np.random.randn(dim_U, r) * 0.1
            V_init = np.random.randn(dim_V, r) * 0.1
            W_init = np.random.randn(dim_W, r) * 0.1
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                sol = least_squares(residual, params_init, args=(r,), method='lm', max_nfev=1000)
                if sol.success and np.max(np.abs(sol.fun)) < 1e-4:
                    U_sol = sol.x[0:r*dim_U].reshape(r, dim_U).T
                    V_sol = sol.x[r*dim_U:r*(dim_U+dim_V)].reshape(r, dim_V).T
                    W_sol = sol.x[r*(dim_U+dim_V):].reshape(r, dim_W).T
                    
                    # Final verification
                    if verify(U_sol, V_sol, W_sol, tol=1e-3):
                        return {
                            "U": U_sol.tolist(),
                            "V": V_sol.tolist(),
                            "W": W_sol.tolist()
                        }
            except Exception:
                continue

    # Fallback: Schoolbook decomposition (Rank 84)
    rank_school = n * m * p
    U = np.zeros((dim_U, rank_school))
    V = np.zeros((dim_V, rank_school))
    W = np.zeros((dim_W, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
