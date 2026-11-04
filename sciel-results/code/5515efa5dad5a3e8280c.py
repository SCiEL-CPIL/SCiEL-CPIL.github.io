import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    # Schoolbook decomposition
    rank_school = n * m * p  # 45
    U_fb = np.zeros((n*m, rank_school), dtype=np.float64)
    V_fb = np.zeros((m*p, rank_school), dtype=np.float64)
    W_fb = np.zeros((p*n, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1.0
                V_fb[j*p+k, col] = 1.0
                W_fb[k*n+i, col] = 1.0
                col += 1

    target = np.ones((n, m, p), dtype=np.float64)

    def verify(U, V, W):
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(result, target, atol=1e-9, rtol=1e-9)

    def error_func(params, R):
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        U = params[:size_u*R].reshape(size_u, R)
        V = params[size_u*R : size_u*R + size_v*R].reshape(size_v, R)
        W = params[size_u*R + size_v*R :].reshape(size_w, R)
        
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        
        result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        diff = result - target
        return np.sum(diff**2)

    def error_grad(params, R):
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        U = params[:size_u*R].reshape(size_u, R)
        V = params[size_u*R : size_u*R + size_v*R].reshape(size_v, R)
        W = params[size_u*R + size_v*R :].reshape(size_w, R)
        
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        
        result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        diff = result - target
        
        # Gradient computation
        # dE/dU_ijk = sum_{k',c} diff_ijk * V_jk'c * W_k'ic  ... wait, indices
        # E = sum_ijk diff_ijk^2
        # dE/dU_ijk = 2 * sum_ijk diff_ijk * (d result_ijk / d U_ijk)
        # result_ijk = sum_c U_ijk_c * V_jkc_c * W_kic_c
        # d result_ijk / d U_ijk_c = V_jkc_c * W_kic_c
        
        # Let's use einsum for gradients
        # dU = 2 * diff * (V * W) contracted over c
        # V is (m, p, R), W is (p, n, R)
        # We need to compute sum_c V[j,k,c] * W[k,i,c] for each i,j,k?
        # No, U is indexed by i,j,c.
        # dE/dU[i,j,c] = 2 * sum_{k} diff[i,j,k] * V[j,k,c] * W[k,i,c]
        
        # Let's compute term by term or vectorized
        # grad_U[i,j,c] = 2 * sum_k diff[i,j,k] * V[j,k,c] * W[k,i,c]
        
        # grad_U shape (n, m, R)
        # diff shape (n, m, p)
        # V shape (m, p, R)
        # W shape (p, n, R)
        
        # grad_U = 2 * einsum('ijk,jkc,kic->ijc', diff, V_r, W_r)
        grad_U = 2 * np.einsum('ijk,jkc,kic->ijc', diff, V_r, W_r)
        
        # grad_V[j,k,c] = 2 * sum_i diff[i,j,k] * U[i,j,c] * W[k,i,c]
        grad_V = 2 * np.einsum('ijk,ijc,kic->jkc', diff, U_r, W_r)
        
        # grad_W[k,i,c] = 2 * sum_j diff[i,j,k] * U[i,j,c] * V[j,k,c]
        grad_W = 2 * np.einsum('ijk,ijc,jkc->kic', diff, U_r, V_r)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return np.sum(diff**2), grad

    def solve_rank(R, max_restarts=50):
        size_u = n * m
        size_v = m * p
        size_w = p * n
        total_size = (size_u + size_v + size_w) * R
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Better initialization: Random normal, scaled
            # Sometimes initializing with schoolbook-like structures helps?
            # But random is standard for tensor decomposition.
            params0 = np.random.randn(total_size) * 0.5
            
            try:
                # Use L-BFGS-B with gradient
                res = minimize(error_func, params0, args=(R,), method='L-BFGS-B', 
                               jac=error_grad,
                               options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14, 'maxfun': 5000})
                
                if res.fun < 1e-8:
                    U = res.x[:size_u*R].reshape(size_u, R)
                    V = res.x[size_u*R : size_u*R + size_v*R].reshape(size_v, R)
                    W = res.x[size_u*R + size_v*R :].reshape(size_w, R)
                    
                    # Snap to nearest 0.5
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    if verify(U_snap, V_snap, W_snap):
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
            except Exception:
                pass
                
        return None

    # Try to find lower ranks
    # Based on R(3,3,2)=13 and R(3,3,3)=23, R(3,3,5) <= 36.
    # Try 36 first, then 37, 38...
    for R in range(36, 45):
        if time.time() > deadline - 10:
            break
        
        result = solve_rank(R, max_restarts=50)
        if result is not None:
            return result

    # Return schoolbook
    return {
        "U": U_fb.tolist(),
        "V": V_fb.tolist(),
        "W": W_fb.tolist()
    }
