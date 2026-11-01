import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    n, m, p = 2, 3, 4
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Create the target tensor T*
    T_target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T_target[alpha, beta, gamma] = 1.0

    def loss(params, r):
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
        return np.sum(error**2)
        
    def grad(params, r):
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
            grad_U[:, c] = 2.0 * np.einsum('abg,b,g->a', error, V[:, c], W[:, c])
            grad_V[:, c] = 2.0 * np.einsum('abg,a,g->b', error, U[:, c], W[:, c])
            grad_W[:, c] = 2.0 * np.einsum('abg,a,b->g', error, U[:, c], V[:, c])
            
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    def snap_and_verify(U, V, W, r):
        U_snapped = np.round(U * 2) / 2
        V_snapped = np.round(V * 2) / 2
        W_snapped = np.round(W * 2) / 2
        
        T_decomp = np.zeros((dim_u, dim_v, dim_w))
        for c in range(r):
            T_decomp += np.einsum('a,b,c->abc', U_snapped[:, c], V_snapped[:, c], W_snapped[:, c])
            
        if np.allclose(T_decomp, T_target, atol=1e-6):
            return U_snapped, V_snapped, W_snapped
        return None

    # Try to find rank 23
    r = 23
    best_loss = np.inf
    best_params = None
    
    # Limit search time to 10 seconds
    search_deadline = start_time + 10.0
    
    for restart in range(50):
        if time.time() > search_deadline:
            break
        if time.time() > deadline:
            break
            
        # Random initialization
        params = np.random.randn(dim_u * r + dim_v * r + dim_w * r) * 0.5
        
        # Use L-BFGS-B
        try:
            res = minimize(loss, params, method='L-BFGS-B', jac=grad, args=(r,), 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
                
            if best_loss < 1e-6:
                break
        except Exception:
            pass

    if best_params is not None and best_loss < 1e-6:
        size_u = dim_u * r
        size_v = dim_v * r
        size_w = dim_w * r
        
        U = best_params[:size_u].reshape((dim_u, r))
        V = best_params[size_u:size_u+size_v].reshape((dim_v, r))
        W = best_params[size_u+size_v:size_u+size_v+size_w].reshape((dim_w, r))
        
        snapped = snap_and_verify(U, V, W, r)
        if snapped is not None:
            U_snapped, V_snapped, W_snapped = snapped
            return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}

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
