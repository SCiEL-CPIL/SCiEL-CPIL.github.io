import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    target_rank = 29
    start_time = time.time()
    deadline = start_time + 140 # 10s buffer

    # Build Tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    def check_decomposition(U, V, W, R):
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.sum((Rec - T)**2)
        if err > 1e-8:
            return False, err
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        if np.allclose(U2, np.round(U2), atol=1e-6) and \
           np.allclose(V2, np.round(V2), atol=1e-6) and \
           np.allclose(W2, np.round(W2), atol=1e-6):
            U_out = [[float(x/2) for x in row] for row in U2]
            V_out = [[float(x/2) for x in row] for row in V2]
            W_out = [[float(x/2) for x in row] for row in W2]
            return True, err
        return False, err

    def objective(params, lam):
        R = target_rank
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:]
        
        U = U_flat.reshape(n*m, R)
        V = V_flat.reshape(m*p, R)
        W = W_flat.reshape(p*n, R)
        
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.sum((Rec - T)**2)
        
        # Penalty for non-half-integers
        # sin^2(pi * 2x) is 0 when 2x is integer
        penalty_U = np.sum(np.sin(np.pi * 2 * U_flat)**2)
        penalty_V = np.sum(np.sin(np.pi * 2 * V_flat)**2)
        penalty_W = np.sum(np.sin(np.pi * 2 * W_flat)**2)
        
        return err + lam * (penalty_U + penalty_V + penalty_W)

    best_params = None
    best_err = float('inf')
    
    # Homotopy path for lambda
    lambdas = [0.0, 1.0, 10.0, 100.0, 1000.0]
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        np.random.seed(42 + restart)
        params = np.random.randn(target_rank * (n*m + m*p + p*n))
        
        current_params = params
        for lam in lambdas:
            if time.time() > deadline:
                break
            try:
                res = minimize(objective, current_params, args=(lam,), method='L-BFGS-B', 
                               options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
                current_params = res.x
            except Exception:
                pass
        
        # Final check with high lambda to ensure integrality
        # Actually, the last lambda in the loop was 1000.
        # Let's do one more polish with very high lambda?
        # Or just check the current_params.
        
        # Snap to half-integers
        R = target_rank
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        U_flat = current_params[:size_U]
        V_flat = current_params[size_U:size_U+size_V]
        W_flat = current_params[size_U+size_V:]
        
        U = (np.round(U_flat * 2) / 2).reshape(n*m, R)
        V = (np.round(V_flat * 2) / 2).reshape(m*p, R)
        W = (np.round(W_flat * 2) / 2).reshape(p*n, R)
        
        valid, err = check_decomposition(U, V, W, R)
        if valid:
            return {"U": U, "V": V, "W": W}
            
        # If not valid, maybe the raw error was low but integrality failed?
        # We can try to optimize with a very high lambda starting from this point
        # But we are out of time maybe.
        
    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
