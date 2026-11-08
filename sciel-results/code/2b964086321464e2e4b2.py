import numpy as np
import time
import random

def try_find_decomp(n, m, p, R, max_iters=500, seed=42):
    np.random.seed(seed)
    # Target tensor
    T = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(m):
                for l in range(p):
                    if j == k:
                        T[i, j, k, l] = 1.0
    
    # Initialize
    U = np.random.randint(-1, 2, (n*m, R))
    V = np.random.randint(-1, 2, (m*p, R))
    W = np.random.randint(-1, 2, (p*n, R))
    
    # Snap to half-integers? Let's stick to integers first, or allow 0.5.
    # The problem allows half-integers.
    # Let's multiply by 2 to work with integers.
    # T' = 2T. Entries 0 or 2.
    # U', V', W' integers.
    # Sum u' v' w' = 2 T.
    
    U = U * 2
    V = V * 2
    W = W * 2
    T_target = T * 2
    
    best_err = np.inf
    best_sol = None
    
    for iter in range(max_iters):
        # Update U
        # T_target[i,j,k,l] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+l, r]
        # For fixed V, W, we have a linear system for U.
        # But it's a sum over r.
        # Let's just do gradient descent on the error squared.
        
        # Error E = T_target - einsum('ir,kr,lr->ikl', U, V, W) # Shapes need care
        # U is (nm, R), V is (mp, R), W is (pn, R)
        # Tensor T is (n, m, m, p)
        # T[i, j, k, l] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+l, r]
        
        # Reshape for einsum
        U_reshaped = U.reshape(n, m, R)
        V_reshaped = V.reshape(m, p, R)
        W_reshaped = W.reshape(p, n, R)
        
        T_current = np.einsum('ijr,jkr,klr->ijkl', U_reshaped, V_reshaped, W_reshaped)
        err = T_target - T_current
        err_norm = np.linalg.norm(err)
        
        if err_norm < 1e-6:
            # Found exact solution (in integers, so original was half-integers)
            return U/2.0, V/2.0, W/2.0
        
        if err_norm < best_err:
            best_err = err_norm
            best_sol = (U.copy(), V.copy(), W.copy())
            
        # Gradient step
        # dE/du = -2 * (T_target - T_current) * v * w
        # This is complex to compute efficiently without loops.
        # Given small sizes, let's just do a simple random perturbation + local search.
        
        # Perturb one element of U, V, or W
        if iter % 3 == 0:
            idx = np.random.randint(n*m)
            r = np.random.randint(R)
            delta = np.random.choice([-2, -1, 1, 2])
            U_new = U.copy()
            U_new[idx, r] += delta
            U_reshaped_new = U_new.reshape(n, m, R)
            T_new = np.einsum('ijr,jkr,klr->ijkl', U_reshaped_new, V_reshaped, W_reshaped)
            err_new = T_target - T_new
            if np.linalg.norm(err_new) < err_norm:
                U = U_new
        elif iter % 3 == 1:
            idx = np.random.randint(m*p)
            r = np.random.randint(R)
            delta = np.random.choice([-2, -1, 1, 2])
            V_new = V.copy()
            V_new[idx, r] += delta
            V_reshaped_new = V_new.reshape(m, p, R)
            T_new = np.einsum('ijr,jkr,klr->ijkl', U_reshaped, V_reshaped_new, W_reshaped)
            err_new = T_target - T_new
            if np.linalg.norm(err_new) < err_norm:
                V = V_new
        else:
            idx = np.random.randint(p*n)
            r = np.random.randint(R)
            delta = np.random.choice([-2, -1, 1, 2])
            W_new = W.copy()
            W_new[idx, r] += delta
            W_reshaped_new = W_new.reshape(p, n, R)
            T_new = np.einsum('ijr,jkr,klr->ijkl', U_reshaped, V_reshaped, W_reshaped_new)
            err_new = T_target - T_new
            if np.linalg.norm(err_new) < err_norm:
                W = W_new
                
    return None
