import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Lower bound is max(n*m, m*p, p*n) = 15
    # Upper bound is n*m*p = 30
    # We try to find the minimal rank
    
    # Precompute indices for the tensor entries
    # The tensor is T[i, j, k] = 1
    # The equation is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # We will iterate R from 15 to 29
    # For each R, we try a few random initializations
    
    def residual(params, R):
        # params is a flat vector of length R * (nm + mp + pn)
        # Split into U, V, W
        size_U = nm * R
        size_V = mp * R
        size_W = pn * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:]
        
        U = U_flat.reshape(nm, R)
        V = V_flat.reshape(mp, R)
        W = W_flat.reshape(pn, R)
        
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    # Sum over r
                    res[i, j, k] = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
        
        # Target is 1
        return (res - 1.0).flatten()

    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    start_time = time.time()
    timeout = 120.0 # seconds
    
    # Try ranks from 15 to 29
    for R in range(15, 30):
        if time.time() - start_time > timeout - 10:
            break
            
        # Try a few random initializations
        for attempt in range(5):
            if time.time() - start_time > timeout - 5:
                break
                
            # Random initialization
            # Scale matters. Let's try uniform [-1, 1] or similar.
            # Since the target is 1, and we have R terms, average term is 1/R.
            # But they can be negative.
            # Let's use a small random initialization to help convergence.
            x0 = np.random.uniform(-1, 1, R * (nm + mp + pn))
            
            try:
                result = least_squares(residual, x0, args=(R,), method='lm', max_nfev=1000)
                
                if result.success or result.cost < 1e-8:
                    # Check if we can snap to half-integers
                    # First, check the float solution
                    U_flat = result.x[:nm*R]
                    V_flat = result.x[nm*R:nm*R+mp*R]
                    W_flat = result.x[nm*R+mp*R:]
                    
                    U = U_flat.reshape(nm, R)
                    V = V_flat.reshape(mp, R)
                    W = W_flat.reshape(pn, R)
                    
                    # Try to round to nearest half-integer
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify rounded solution
                    res_rounded = np.zeros((n, m, p))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx_u = i * m + j
                                idx_v = j * p + k
                                idx_w = k * n + i
                                res_rounded[i, j, k] = np.dot(U_rounded[idx_u, :], V_rounded[idx_v, :] * W_rounded[idx_w, :])
                    
                    if np.allclose(res_rounded, 1.0, atol=1e-6):
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                    else:
                        # If rounding fails, check if the float solution is close enough
                        # and if the problem allows floats? "integer/half-integer entries"
                        # This implies we MUST return integers or half-integers.
                        # If we can't find a half-integer solution, we should not return the float one.
                        pass
                        
            except Exception:
                continue
                
    # If no better rank found, return schoolbook
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
                r += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
