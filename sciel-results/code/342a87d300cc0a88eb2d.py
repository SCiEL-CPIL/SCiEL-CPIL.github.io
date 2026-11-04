import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Schoolbook decomposition as guaranteed fallback
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

    # Target tensor: T[i,j,k] = 1 for all valid i,j,k
    target = np.ones((n, m, p), dtype=np.float64)

    def verify(U, V, W):
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(result, target, atol=1e-9, rtol=1e-9)

    def error_and_grad(params, R):
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
        
        error = np.sum(diff**2)
        
        # Gradients
        grad_U = 2 * np.einsum('ijk,jkc,kic->ijc', diff, V_r, W_r)
        grad_V = 2 * np.einsum('ijk,ijc,kic->jkc', diff, U_r, W_r)
        grad_W = 2 * np.einsum('ijk,ijc,jkc->kic', diff, U_r, V_r)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return error, grad

    def snap_and_verify(U, V, W, tolerance=0.25):
        """Snap to nearest 0.5 and verify"""
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        if verify(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap, True
        
        # Try integer snapping
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if verify(U_int, V_int, W_int):
            return U_int, V_int, W_int, True
        
        return None, None, None, False

    def solve_rank(R, max_restarts=30):
        """Try to find a rank-R decomposition using structured initializations"""
        size_u = n * m
        size_v = m * p
        size_w = p * n
        total_size = (size_u + size_v + size_w) * R
        
        best_error = np.inf
        best_params = None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Different initialization strategies
            if restart % 3 == 0:
                # Random normal
                params0 = np.random.randn(total_size) * 0.3
            elif restart % 3 == 1:
                # Sparse random - many zeros
                params0 = np.random.choice([-1, 0, 1], size=total_size) * 0.5
            else:
                # Structured: initialize with partial schoolbook pattern
                params0 = np.zeros(total_size)
                # Add some structure based on known patterns
                for c in range(R):
                    i = c % n
                    j = (c // n) % m
                    k = (c // (n*m)) % p
                    if i*m+j < size_u:
                        params0[(i*m+j)*R + c] = 1.0
                    if j*p+k < size_v:
                        params0[size_u*R + (j*p+k)*R + c] = 1.0
                    if k*n+i < size_w:
                        params0[(size_u+size_v)*R + (k*n+i)*R + c] = 1.0
                # Add random perturbation
                params0 += np.random.randn(total_size) * 0.1
            
            # Use Nelder-Mead for small problems, or L-BFGS-B
            from scipy.optimize import minimize
            
            try:
                res = minimize(
                    lambda x: error_and_grad(x, R)[0],
                    params0,
                    jac=lambda x: error_and_grad(x, R)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000}
                )
                
                if res.fun < best_error:
                    best_error = res.fun
                    best_params = res.x.copy()
                
                if res.fun < 1e-10:
                    U = res.x[:size_u*R].reshape(size_u, R)
                    V = res.x[size_u*R : size_u*R + size_v*R].reshape(size_v, R)
                    W = res.x[size_u*R + size_v*R :].reshape(size_w, R)
                    
                    U_snap, V_snap, W_snap, success = snap_and_verify(U, V, W)
                    if success:
                        return U_snap, V_snap, W_snap
                        
            except Exception:
                pass
        
        # If we got close, try to polish
        if best_params is not None and best_error < 1e-6:
            U = best_params[:size_u*R].reshape(size_u, R)
            V = best_params[size_u*R : size_u*R + size_v*R].reshape(size_v, R)
            W = best_params[size_u*R + size_v*R :].reshape(size_w, R)
            
            U_snap, V_snap, W_snap, success = snap_and_verify(U, V, W)
            if success:
                return U_snap, V_snap, W_snap
        
        return None

    # Try ranks from low to high, starting with promising values
    # Based on known results: R(3,3,3)=23, so R(3,3,5) should be around 30-38
    target_ranks = [28, 30, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
    
    for R in target_ranks:
        if time.time() > deadline - 15:
            break
        
        result = solve_rank(R, max_restarts=20)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Return schoolbook as guaranteed fallback
    return {
        "U": U_fb.tolist(),
        "V": V_fb.tolist(),
        "W": W_fb.tolist()
    }
