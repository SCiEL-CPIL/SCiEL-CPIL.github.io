import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    n, m, p = 3, 3, 5
    
    # Target tensor T[I, J, K]
    # I = i*m + j (0..8)
    # J = j'*p + k (0..14)
    # K = k'*n + i' (0..14)
    # T[I, J, K] = 1 if i=i', j=j', k=k'
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    def check_exactness(U, V, W, tol=1e-6):
        # Round to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if np.max(np.abs(U - U_r)) > 1e-3 or np.max(np.abs(V - V_r)) > 1e-3 or np.max(np.abs(W - W_r)) > 1e-3:
            return False, None, None, None
            
        R = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r) - T
        if np.max(np.abs(R)) < tol:
            return True, U_r, V_r, W_r
        return False, None, None, None

    # Strategy: Rank 39 = 21 (core 3x3x3) + 18 (extra 3x3x2)
    # We fix the extra part (identity for k=3,4) and optimize the core part.
    
    r_core = 21
    r_extra = 18
    r_total = r_core + r_extra
    
    # Fixed extra terms
    U_extra = np.zeros((n*m, r_extra))
    V_extra = np.zeros((m*p, r_extra))
    W_extra = np.zeros((p*n, r_extra))
    
    col = 0
    for i in range(n):
        for j in range(m):
            U_extra[i*m + j, col] = 1.0
            V_extra[j*p + 3, col] = 1.0
            W_extra[3*n + i, col] = 1.0
            col += 1
            U_extra[i*m + j, col] = 1.0
            V_extra[j*p + 4, col] = 1.0
            W_extra[4*n + i, col] = 1.0
            col += 1

    # Masks for core optimization
    # Core terms should ideally only use J with k in {0,1,2} and K with k' in {0,1,2}
    # But to be safe, we allow full support in optimization, but initialize with masked support.
    
    mask_V_core = np.zeros(m*p)
    for j in range(m):
        for k in range(3):
            mask_V_core[j*p + k] = 1.0
            
    mask_W_core = np.zeros(p*n)
    for k in range(3):
        for i in range(n):
            mask_W_core[k*n + i] = 1.0

    T_core = np.zeros((n*m, m*p, p*n))
    T_core[:, mask_V_core.astype(bool), :] = T[:, mask_V_core.astype(bool), :]
    for k in range(3, p):
        T_core[:, :, k*n:(k+1)*n] = 0.0

    def compute_core_residual(U_c, V_c, W_c):
        return np.einsum('ic,jc,kc->ijk', U_c, V_c, W_c) - T_core

    np.random.seed(42)
    
    # Try to find rank 21 decomposition for the core
    found = False
    best_rank = 45
    best_U, best_V, best_W = None, None, None

    for attempt in range(50):
        if time.time() > deadline - 10:
            break
            
        r = r_core
        U_c = np.random.randn(n*m, r) * 0.5
        V_c = np.random.randn(m*p, r) * 0.5
        W_c = np.random.randn(p*n, r) * 0.5
        
        # Initialize with masked support to encourage block structure
        V_c *= mask_V_core[:, None]
        W_c *= mask_W_core[:, None]
        
        try:
            from scipy.optimize import minimize
            
            def objective(params):
                U_t = params[:n*m*r].reshape(n*m, r)
                V_t = params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W_t = params[n*m*r+m*p*r:].reshape(p*n, r)
                
                # Enforce mask softly or hard? Hard is better for structure.
                V_t = V_t * mask_V_core[:, None]
                W_t = W_t * mask_W_core[:, None]
                
                R = compute_core_residual(U_t, V_t, W_t)
                return np.sum(R**2)
            
            params0 = np.concatenate([U_c.flatten(), V_c.flatten(), W_c.flatten()])
            
            # Use Powell for derivative-free robustness
            res = minimize(
                objective,
                params0,
                method='Powell',
                options={'maxiter': 500, 'maxfev': 5000}
            )
            
            if res.fun < 1e-4:
                U_t = res.x[:n*m*r].reshape(n*m, r)
                V_t = res.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W_t = res.x[n*m*r+m*p*r:].reshape(p*n, r)
                
                # Assemble full
                U_full = np.hstack([U_t, U_extra])
                V_full = np.hstack([V_t, V_extra])
                W_full = np.hstack([W_t, W_extra])
                
                is_exact, U_r, V_r, W_r = check_exactness(U_full, V_full, W_full)
                if is_exact:
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
        except Exception:
            pass

    # Fallback: Rank 45 (Schoolbook)
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, column] = 1
                V[j*p + k, column] = 1
                W[k*n + i, column] = 1
                column += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
