import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
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
    
    # Build Tensor T
    # T[a, b, c] = 1 if i_a==i_c, j_a==j_b, k_b==k_c
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Function to check if a decomposition is valid
    def check_decomposition(U, V, W, R):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Rec[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.sum((Rec - T)**2)
        if err > 1e-8:
            return False, err
        # Check if entries are close to half-integers
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        if np.allclose(U2, np.round(U2), atol=1e-6) and \
           np.allclose(V2, np.round(V2), atol=1e-6) and \
           np.allclose(W2, np.round(W2), atol=1e-6):
            # Convert to list of lists of float (half-integers)
            U_out = [[float(x/2) for x in row] for row in U2]
            V_out = [[float(x/2) for x in row] for row in V2]
            W_out = [[float(x/2) for x in row] for row in W2]
            return True, err
        return False, err

    # We will try to find a rank 29 decomposition.
    # The known rank for 2x3x5 is 29 (or possibly 30? Let's check literature or assume 29 is a good target).
    # Actually, Strassen's algorithm gives rank 7 for 2x2.
    # For 2x3x5, the schoolbook is 30.
    # Let's try to optimize for rank 29.
    
    target_rank = 29
    
    def objective(params):
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
        return err

    best_err = float('inf')
    best_params = None
    
    # Try a few random restarts
    # We need to be careful with time.
    # L-BFGS-B can be slow. Let's try a limited number of restarts.
    
    np.random.seed(42)
    
    # Initial guess: start with schoolbook rank 30 and try to merge two terms?
    # Or just random start.
    
    for restart in range(5):
        if time.time() > deadline:
            break
            
        # Random init
        # Scale matters. Let's use small random values.
        params = np.random.randn(target_rank * (n*m + m*p + p*n)) * 0.1
        
        try:
            from scipy.optimize import minimize
            # Use L-BFGS-B
            res = minimize(objective, params, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_err:
                best_err = res.fun
                best_params = res.x
                # print(f"Restart {restart}: err={best_err}")
        except Exception as e:
            pass

    if best_params is not None and best_err < 1e-6:
        R = target_rank
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        U_flat = best_params[:size_U]
        V_flat = best_params[size_U:size_U+size_V]
        W_flat = best_params[size_U+size_V:]
        
        U = U_flat.reshape(n*m, R)
        V = V_flat.reshape(m*p, R)
        W = W_flat.reshape(p*n, R)
        
        valid, err = check_decomposition(U, V, W, R)
        if valid:
            return {"U": U, "V": V, "W": W}
    
    # If rank 29 failed, try rank 30? No, schoolbook is 30.
    # But maybe we can find a better rank 30?
    # The task asks for rank-minimal.
    # If we can't find 29, we return schoolbook (30).
    
    # Is there a known rank 29 decomposition?
    # Let's check if we can do a smarter search.
    # The tensor has 30 non-zero entries.
    # Rank 29 means one linear dependence.
    
    # For now, if optimization fails, return schoolbook.
    # But we can try to improve the optimization.
    
    # Let's try a different approach:
    # Fix U and V to be sparse or specific structures?
    # Or use a grid search for small integers?
    # The entries are half-integers.
    
    # Let's try one more thing: 
    # If we fail to find rank 29, we return the schoolbook decomposition.
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
