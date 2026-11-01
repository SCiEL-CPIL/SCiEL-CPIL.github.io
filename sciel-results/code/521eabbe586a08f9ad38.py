import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = 2, 3, 4
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Create the target tensor T*
    # T* is a 3D array of shape (dim_u, dim_v, dim_w)
    T_target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T_target[alpha, beta, gamma] = 1.0

    # Flatten target tensor for easier handling in loss function
    # The loss function will compute the decomposed tensor and compare
    # We need to iterate over all alpha, beta, gamma
    
    def loss(params, r):
        """
        params: flattened array of U, V, W
        r: rank
        """
        # Unpack
        size_u = dim_u * r
        size_v = dim_v * r
        size_w = dim_w * r
        
        U_flat = params[:size_u]
        V_flat = params[size_u:size_u+size_v]
        W_flat = params[size_u+size_v:size_u+size_v+size_w]
        
        U = U_flat.reshape((dim_u, r))
        V = V_flat.reshape((dim_v, r))
        W = W_flat.reshape((dim_w, r))
        
        # Compute decomposed tensor
        # T_decomp[alpha, beta, gamma] = sum_c U[alpha, c] * V[beta, c] * W[gamma, c]
        # This is expensive to compute fully for every step.
        # We can compute the error only on the non-zero entries of T_target?
        # No, we must ensure the zero entries are also zero.
        # But the number of entries is dim_u * dim_v * dim_w = 6 * 12 * 8 = 576.
        # This is small. We can compute the full tensor.
        
        # Efficient computation:
        # T_decomp = sum_c U[:, c] @ V[:, c].T @ W[:, c].T ? No.
        # T_decomp[alpha, beta, gamma] = dot(U[alpha], V[beta] * W[gamma])? No.
        # T_decomp[alpha, beta, gamma] = sum_c U[alpha, c] V[beta, c] W[gamma, c]
        
        # We can vectorize this:
        # For each c, U[:, c] is a vector u_c, V[:, c] is v_c, W[:, c] is w_c.
        # The contribution is outer(outer(u_c, v_c), w_c)
        # T_decomp = sum_c np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        
        T_decomp = np.zeros((dim_u, dim_v, dim_w))
        for c in range(r):
            T_decomp += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
            
        error = T_decomp - T_target
        return np.sum(error**2)
        
    def grad(params, r):
        # Numerical gradient or analytical?
        # Analytical is better.
        # Let f = sum (T_decomp - T_target)^2
        # df/dU[a, c] = 2 * sum_{b,g} (T_decomp[a,b,g] - T_target[a,b,g]) * V[b,c] * W[g,c]
        
        size_u = dim_u * r
        size_v = dim_v * r
        size_w = dim_w * r
        
        U_flat = params[:size_u]
        V_flat = params[size_u:size_u+size_v]
        W_flat = params[size_u+size_v:size_u+size_v+size_w]
        
        U = U_flat.reshape((dim_u, r))
        V = V_flat.reshape((dim_v, r))
        W = W_flat.reshape((dim_w, r))
        
        T_decomp = np.zeros((dim_u, dim_v, dim_w))
        for c in range(r):
            T_decomp += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
            
        error = T_decomp - T_target
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for c in range(r):
            # grad_U[a, c] = 2 * sum_{b,g} error[a,b,g] * V[b,c] * W[g,c]
            # This is 2 * (error * outer(V[:, c], W[:, c])).sum(axis=(1,2))
            # Let's compute it
            # error is (dim_u, dim_v, dim_w)
            # We need to contract error with V[:, c] and W[:, c]
            
            # Using einsum:
            # grad_U[:, c] = 2 * einsum('abg, b, g -> a', error, V[:, c], W[:, c])
            grad_U[:, c] = 2.0 * np.einsum('abg,b,g->a', error, V[:, c], W[:, c])
            grad_V[:, c] = 2.0 * np.einsum('abg,a,g->b', error, U[:, c], W[:, c])
            grad_W[:, c] = 2.0 * np.einsum('abg,a,b->g', error, U[:, c], V[:, c])
            
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    # Try to find rank 23
    r = 23
    best_loss = np.inf
    best_params = None
    
    # Initialize with random values
    # Or initialize with schoolbook and drop one column?
    # Schoolbook has rank 24.
    # If we take the schoolbook U, V, W and set the last column to 0, the loss will be high.
    # Better to start random.
    
    for restart in range(5):
        if time.time() > deadline - 20:
            break
            
        # Random initialization
        params = np.random.randn(dim_u * r + dim_v * r + dim_w * r) * 0.1
        
        # Use L-BFGS-B
        res = minimize(loss, params, method='L-BFGS-B', jac=grad, args=(r,), options={'maxiter': 500, 'ftol': 1e-10})
        
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x
            
        if best_loss < 1e-6:
            break
            
    if best_params is not None and best_loss < 1e-6:
        # Extract U, V, W
        size_u = dim_u * r
        size_v = dim_v * r
        size_w = dim_w * r
        
        U = best_params[:size_u].reshape((dim_u, r))
        V = best_params[size_u:size_u+size_v].reshape((dim_v, r))
        W = best_params[size_u+size_v:size_u+size_v+size_w].reshape((dim_w, r))
        
        # Snap to half-integers?
        # The problem asks for integer/half-integer entries.
        # Numerical solutions might not be exactly half-integers.
        # We should snap them.
        U_snapped = np.round(U * 2) / 2
        V_snapped = np.round(V * 2) / 2
        W_snapped = np.round(W * 2) / 2
        
        # Verify snapped solution
        T_decomp = np.zeros((dim_u, dim_v, dim_w))
        for c in range(r):
            T_decomp += np.einsum('a,b,c->abc', U_snapped[:, c], V_snapped[:, c], W_snapped[:, c])
            
        if np.allclose(T_decomp, T_target, atol=1e-6):
            return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
        else:
            # If snapping fails, return the un-snapped one? 
            # The prompt says "integer/half-integer entries". 
            # If I return floats, it might be invalid.
            # But if the numerical solution is close to half-integers, snapping should work.
            # If not, we might have to return schoolbook.
            pass

    # Fallback to schoolbook
    r_schoolbook = 24
    U = np.zeros((dim_u, r_schoolbook))
    V = np.zeros((dim_v, r_schoolbook))
    W = np.zeros((dim_w, r_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
