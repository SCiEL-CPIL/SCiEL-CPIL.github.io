import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    # Precompute target tensor indices
    # T[alpha, beta, gamma] = 1 if valid, else 0
    # alpha = i*m + j
    # beta  = j*p + k
    # gamma = k*n + i
    
    # We don't need the full tensor object, just the non-zero indices
    # for the residual calculation? 
    # Actually, we need the full tensor because we must satisfy the ZERO constraints too.
    # If we only fit the 1s, the 0s will be garbage.
    # So we need the full 900-element vector.
    
    # Create the full target tensor T
    T = np.zeros((size_U, size_V, size_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    T_flat = T.flatten()
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Schoolbook fallback
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(size_U)]
        V = [[0] * rank for _ in range(size_V)]
        W = [[0] * rank for _ in range(size_W)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Residual function for least_squares
    # x is flattened [U, V, W]
    def residuals(x):
        U = x[:size_U * (len(x) // (size_U + size_V + size_W))].reshape(size_U, -1)
        # This reshape logic is fragile if x length changes. 
        # Better to pass R as an argument or use a closure.
        # Let's use a closure for R.
        pass

    # We will define the residual function inside the loop for each R
    # to capture R correctly.
    
    # Try ranks from lower bound up to schoolbook - 1
    # Lower bound is 15.
    for R in range(15, 30):
        if time.time() > deadline:
            break
            
        size_U_R = size_U
        size_V_R = size_V
        size_W_R = size_W
        num_vars = (size_U_R + size_V_R + size_W_R) * R
        
        def make_residual_func(R):
            def residuals(x):
                # Unpack
                U = x[:size_U_R * R].reshape(size_U_R, R)
                V = x[size_U_R * R : (size_U_R + size_V_R) * R].reshape(size_V_R, R)
                W = x[(size_U_R + size_V_R) * R :].reshape(size_W_R, R)
                
                # Compute T_recon
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                
                # Residual
                return (T_recon.flatten() - T_flat)
            return residuals
        
        residual_func = make_residual_func(R)
        
        # Number of restarts
        # We have ~110s. 
        # Let's say we spend 5s per rank on average? 15 ranks * 5s = 75s.
        # So we can afford a few restarts per rank.
        # Let's try 10 restarts for each R.
        num_restarts = 10
        
        for _ in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            # Scale: entries should be around 1.
            # If we use too large numbers, the problem is ill-conditioned.
            # If too small, it might converge to 0.
            # Let's use uniform [-1, 1] or normal.
            x0 = np.random.uniform(-1, 1, num_vars)
            
            try:
                # Use 'trf' method which is robust
                # max_nfev: limit function evaluations to prevent hanging
                res = least_squares(residual_func, x0, method='trf', max_nfev=100)
                
                # Check residual
                if res.cost < 1e-10:
                    # Potential solution
                    x_sol = res.x
                    U = x_sol[:size_U_R * R].reshape(size_U_R, R)
                    V = x_sol[size_U_R * R : (size_U_R + size_V_R) * R].reshape(size_V_R, R)
                    W = x_sol[(size_U_R + size_V_R) * R :].reshape(size_W_R, R)
                    
                    # Snap to half-integers
                    U_snapped = np.round(U * 2) / 2.0
                    V_snapped = np.round(V * 2) / 2.0
                    W_snapped = np.round(W * 2) / 2.0
                    
                    # Verify exactness with snapped values
                    T_recon = np.einsum('ar,br,cr->abc', U_snapped, V_snapped, W_snapped)
                    if np.allclose(T_recon, T, atol=1e-9):
                        return {
                            "U": U_snapped.tolist(),
                            "V": V_snapped.tolist(),
                            "W": W_snapped.tolist()
                        }
                        
            except Exception:
                pass

    # If no lower rank found, return schoolbook
    return get_schoolbook()

# EVOLVE-BLOCK-END
