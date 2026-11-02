import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    target_rank = n * m * p
    # Schoolbook
    U_school = np.zeros((n*m, target_rank))
    V_school = np.zeros((m*p, target_rank))
    W_school = np.zeros((p*n, target_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = target_rank
    best_decomp = (U_school, V_school, W_school)
    
    start_time = time.time()
    deadline = start_time + 120 # 120s safety margin for 150s limit
    
    # Try to find lower ranks
    for R in range(target_rank - 1, 20, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for this rank
        found = False
        for attempt in range(50): # Limit attempts to avoid long hangs
             if time.time() > deadline:
                 break
             
             # Initialize
             U = np.random.rand(n*m, R) * 2 - 1
             V = np.random.rand(m*p, R) * 2 - 1
             W = np.random.rand(p*n, R) * 2 - 1
             
             # Optimization
             # We want to minimize || T - sum(u_c v_c w_c) ||^2
             # T is sparse.
             
             # Simple gradient descent might be slow. Let's use a more direct approach.
             # Since the target is very small, we can check if the system of equations
             # has a solution.
             
             # Actually, for small R, we can use the fact that the tensor is determined by its entries.
             # The condition is that for all i,j,k: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = T[i,j,k]
             
             # This is a set of bilinear/cubic equations.
             # Let's stick to numerical optimization.
             
             def loss(params):
                 U_flat = params[:n*m*R].reshape(n*m, R)
                 V_flat = params[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                 W_flat = params[n*m*R + m*p*R :].reshape(p*n, R)
                 
                 # Compute reconstruction
                 # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
                 U_r = U_flat.reshape(n, m, R)
                 V_r = V_flat.reshape(m, p, R)
                 W_r = W_flat.reshape(p, n, R)
                 
                 # Sum over R
                 # Result[i, j, k] = sum_c U_r[i, j, c] V_r[j, k, c] W_r[k, i, c]
                 # This is a contraction.
                 # We can use einsum: 'ijc,jkc,kic->ijk'
                 rec = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                 
                 # Target
                 # T[i,j,k] is 1 if (i,j,k) is in the support, else 0.
                 # Support: all i in 0..n-1, j in 0..m-1, k in 0..p-1
                 # So T is all 1s.
                 target = np.ones((n, m, p))
                 
                 return np.sum((rec - target)**2)
             
             # Use scipy minimize
             from scipy.optimize import minimize
             x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
             
             res = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
             
             if res.fun < 1e-8:
                 # Check for half-integer rounding
                 U_opt = res.x[:n*m*R].reshape(n*m, R)
                 V_opt = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                 W_opt = res.x[n*m*R + m*p*R :].reshape(p*n, R)
                 
                 # Round to nearest 0.5
                 U_int = np.round(U_opt * 2) / 2
                 V_int = np.round(V_opt * 2) / 2
                 W_int = np.round(W_opt * 2) / 2
                 
                 # Verify exactness
                 U_r = U_int.reshape(n, m, R)
                 V_r = V_int.reshape(m, p, R)
                 W_r = W_int.reshape(p, n, R)
                 rec = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                 
                 if np.allclose(rec, np.ones((n, m, p)), atol=1e-9):
                     # Found a valid decomposition
                     best_rank = R
                     best_decomp = (U_int, V_int, W_int)
                     found = True
                     break
                     
        if found:
            break # Found a valid lower rank, stop searching
            
    # Convert best_decomp to lists
    U, V, W = best_decomp
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
