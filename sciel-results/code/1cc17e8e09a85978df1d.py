import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor is all 1s of shape (n, m, p)
    # T[i, j, k] = 1
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Lower bound for rank
    # Rank of matricization (n*m, p) is min(n*m, p) = 8
    # Rank of matricization (n, m*p) is min(n, m*p) = 3
    # Rank of matricization (m, n*p) is min(m, n*p) = 4
    # So r >= 8.
    # But likely higher.
    
    for r in range(10, 60):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W
            # Shapes: U (n*m, r), V (m*p, r), W (p*n, r)
            U_init = np.random.randn(n * m, r) * 0.1
            V_init = np.random.randn(m * p, r) * 0.1
            W_init = np.random.randn(p * n, r) * 0.1
            
            # Flatten parameters
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(x):
                U = x[:n*m*r].reshape(n*m, r)
                V = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W = x[n*m*r + m*p*r :].reshape(p*n, r)
                
                # Compute tensor T = sum_k U[:, k] (outer) V[:, k] (outer) W[:, k]
                # T[i, j, k] = sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
                
                # Efficient computation:
                # T_ijk = sum_l U_imj_l V_jpk_l W_kni_l
                # This is a 3-way contraction.
                
                # We can reshape and use einsum
                # U: (n, m, r)
                # V: (m, p, r)
                # W: (p, n, r)
                
                U_3d = U.reshape(n, m, r)
                V_3d = V.reshape(m, p, r)
                W_3d = W.reshape(p, n, r)
                
                # T[i, j, k] = sum_l U_3d[i, j, l] * V_3d[j, k, l] * W_3d[k, i, l]
                # Note indices:
                # U index: i, j
                # V index: j, k
                # W index: k, i
                
                # Let's use einsum
                # 'ijl,jkl,kil->ijk'
                T = np.einsum('ijl,jkl,kil->ijk', U_3d, V_3d, W_3d, optimize=True)
                
                # Target is all 1s
                # Error = sum (T - 1)^2
                error = np.sum((T - 1.0)**2)
                return error
            
            # Run optimization
            # L-BFGS-B is good for medium scale
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-12})
            
            if res.fun < 1e-6:
                # Extract and round
                U_opt = res.x[:n*m*r].reshape(n*m, r)
                V_opt = res.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W_opt = res.x[n*m*r + m*p*r :].reshape(p*n, r)
                
                # Round to nearest 0.5
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                # Verify exactly
                U_3d = U_rounded.reshape(n, m, r)
                V_3d = V_rounded.reshape(m, p, r)
                W_3d = W_rounded.reshape(p, n, r)
                
                T_check = np.einsum('ijl,jkl,kil->ijk', U_3d, V_3d, W_3d, optimize=True)
                
                if np.allclose(T_check, 1.0, atol=1e-9):
                    # Convert to lists of lists
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
    # Fallback to schoolbook if search fails
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
# EVOLVE-BLOCK-END
