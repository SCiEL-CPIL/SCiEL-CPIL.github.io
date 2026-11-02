import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    # Target ranks to try, from low to high
    # Lower bound is 21. Schoolbook is 63.
    target_ranks = [25, 30, 35, 40, 45, 50, 55, 60]
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def tensor_error(params, R):
        # Reshape params into U, V, W
        # U: (num_u, R), V: (num_v, R), W: (num_w, R)
        # Total params: R * (num_u + num_v + num_w)
        size_u = num_u * R
        size_v = num_v * R
        size_w = num_w * R
        
        u_flat = params[:size_u]
        v_flat = params[size_u:size_u+size_v]
        w_flat = params[size_u+size_v:]
        
        U = u_flat.reshape((num_u, R))
        V = v_flat.reshape((num_v, R))
        W = w_flat.reshape((num_w, R))
        
        # Compute T = sum_r U[:,r] outer V[:,r] outer W[:,r]
        # T shape: (num_u, num_v, num_w)
        # Efficient computation:
        # T = np.einsum('ir,jr,kr->ijk', U, V, W)
        T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
        
        # Target tensor T_target
        # T_target[i*m+j, j'*p+k, k'*n+i] = 1 if j==j' and i==i' and k==k' ?
        # Wait, the indices in the problem description:
        # U index: i*m+j
        # V index: j'*p+k
        # W index: k'*n+i
        # The product is sum_r U[i,j,r] V[j',k,r] W[k',i',r]
        # This should equal 1 if the indices correspond to the multiplication.
        # Multiplication: C[i, k] = sum_j A[i, j] B[j, k]
        # So the term A[i, j] B[j, k] contributes to C[i, k].
        # In the tensor T, the entry T[a, b, c] is 1 if a=(i,j), b=(j,k), c=(k,i) for the same i,j,k.
        # Wait, the W index is k*n+i. So c = k*n+i.
        # So T[a, b, c] = 1 if there exist i,j,k such that:
        # a = i*m + j
        # b = j*p + k
        # c = k*n + i
        # And for all other combinations, T is 0.
        
        # We can precompute T_target once.
        # But inside the function, we just compute the error.
        # To avoid recomputing T_target, we can pass it or use a closure.
        
        # Error: sum (T_approx - T_target)^2
        err = np.sum((T_approx - T_target)**2)
        return err

    # Precompute T_target
    T_target = np.zeros((num_u, num_v, num_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    best_result = None
    best_rank = n * m * p # Schoolbook rank
    
    # Fallback: Schoolbook
    U_school = np.zeros((num_u, best_rank))
    V_school = np.zeros((num_v, best_rank))
    W_school = np.zeros((num_w, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    for R in target_ranks:
        if time.time() > deadline:
            break
            
        # Initialize with random values
        np.random.seed(42)
        params_init = np.random.randn(R * (num_u + num_v + num_w))
        
        # Define objective
        def objective(params):
            return tensor_error(params, R)
            
        # Try to optimize
        # Use L-BFGS-B
        try:
            res = minimize(objective, params_init, method='L-BFGS-B', 
                           options={'maxiter': 100, 'maxls': 10})
        except Exception:
            continue
            
        if res.success and res.fun < 1e-6:
            # Check if we can round to half-integers
            size_u = num_u * R
            size_v = num_v * R
            size_w = num_w * R
            
            u_flat = res.x[:size_u]
            v_flat = res.x[size_u:size_u+size_v]
            w_flat = res.x[size_u+size_v:]
            
            U_opt = u_flat.reshape((num_u, R))
            V_opt = v_flat.reshape((num_v, R))
            W_opt = w_flat.reshape((num_w, R))
            
            # Round to nearest 0.5
            U_rounded = np.round(U_opt * 2) / 2
            V_rounded = np.round(V_opt * 2) / 2
            W_rounded = np.round(W_opt * 2) / 2
            
            # Check exactness
            T_check = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
            if np.allclose(T_check, T_target, atol=1e-9):
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                
    # If no better solution found, return schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
