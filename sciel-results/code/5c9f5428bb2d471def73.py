import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    timeout = 110  # Safety margin for 120s limit (evaluator kills at 150s, but 120s is the task limit)
    
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor T[i, j, k] is 1 for all i,j,k.
    # We seek U, V, W of rank R such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    def get_tensor_flat(U, V, W, R):
        """
        Compute the tensor T[i,j,k] from factors U, V, W.
        T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        """
        T = np.zeros((n, m, p))
        for r in range(R):
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            # 'ij,jk,ki->ijk'
            T += np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r)
        return T

    def objective(x, R):
        U_curr = x[:n*m*R].reshape(n*m, R)
        V_curr = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
        W_curr = x[n*m*R + m*p*R :].reshape(p*n, R)
        
        T = get_tensor_flat(U_curr, V_curr, W_curr, R)
        T_target = np.ones((n, m, p))
        res = T - T_target
        return np.sum(res**2)

    # 1. Initialize with Schoolbook Solution (Rank 125)
    rank_init = n * m * p
    U_init = np.zeros((n * m, rank_init))
    V_init = np.zeros((m * p, rank_init))
    W_init = np.zeros((p * n, rank_init))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_init[i * m + j, col] = 1.0
                V_init[j * p + k, col] = 1.0
                W_init[k * n + i, col] = 1.0
                col += 1
                
    best_rank = rank_init
    best_U = U_init
    best_V = V_init
    best_W = W_init
    
    # Time check
    if time.time() - start_time > timeout:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
    # 2. Attempt to reduce rank to 124
    # Strategy: Take the first 124 columns of the schoolbook solution, add noise, and optimize.
    # The missing term is the 125th term (i=4, j=4, k=4 in the loop order, which is col=124).
    # By perturbing the remaining 124 terms, we hope they can "absorb" the missing contribution.
    
    R_target = 124
    U_opt = U_init[:, :R_target].copy()
    V_opt = V_init[:, :R_target].copy()
    W_opt = W_init[:, :R_target].copy()
    
    # Add small noise to break symmetry and allow the optimizer to move
    np.random.seed(42)
    U_opt += np.random.normal(0, 0.1, U_opt.shape)
    V_opt += np.random.normal(0, 0.1, V_opt.shape)
    W_opt += np.random.normal(0, 0.1, W_opt.shape)
    
    x0 = np.concatenate([U_opt.flatten(), V_opt.flatten(), W_opt.flatten()])
    
    try:
        if time.time() - start_time < timeout - 5:
            # Use L-BFGS-B. It handles many variables reasonably well if the function is smooth.
            # We restrict iterations to ensure we don't time out.
            result = minimize(objective, x0, args=(R_target,), method='L-BFGS-B', 
                              options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if result.fun < 1e-10:
                # Check if it's truly valid
                U_final = result.x[:n*m*R_target].reshape(n*m, R_target)
                V_final = result.x[n*m*R_target : n*m*R_target + m*p*R_target].reshape(m*p, R_target)
                W_final = result.x[n*m*R_target + m*p*R_target :].reshape(p*n, R_target)
                
                T_final = get_tensor_flat(U_final, V_final, W_final, R_target)
                T_target = np.ones((n, m, p))
                
                if np.linalg.norm(T_final - T_target) < 1e-9:
                    # The problem requires integer/half-integer entries.
                    # Check if the solution is close to such values.
                    # If the optimizer found a valid real solution, it might not be half-integers.
                    # However, for tensor decompositions of this type, solutions often lie in a lattice.
                    # Let's try rounding to nearest 0.5.
                    
                    U_rounded = np.round(U_final * 2) / 2
                    V_rounded = np.round(V_final * 2) / 2
                    W_rounded = np.round(W_final * 2) / 2
                    
                    # Re-check validity after rounding
                    T_rounded = get_tensor_flat(U_rounded, V_rounded, W_rounded, R_target)
                    if np.linalg.norm(T_rounded - T_target) < 1e-9:
                        best_rank = R_target
                        best_U = U_rounded
                        best_V = V_rounded
                        best_W = W_rounded
                    else:
                        # If rounding breaks it, we cannot use this solution as per contract.
                        # We stick to the schoolbook solution which is guaranteed valid.
                        pass
                        
    except Exception:
        pass

    # If we didn't find a better rank, return the schoolbook solution.
    # The schoolbook solution has integer entries.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
