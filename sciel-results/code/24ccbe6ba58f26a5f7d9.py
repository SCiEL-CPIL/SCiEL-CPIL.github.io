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
    
    def compute_residual(U, V, W):
        # U, V, W are flat arrays of length R * nm, etc.
        R = U.shape[0] // nm
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        
        # einsum: 'ijr,jkr,kin->ijk'
        T = np.einsum('ijr,jkr,kin->ijk', U3, V3, W3, optimize=True)
        return T - 1.0

    def residual(params, R):
        size_U = nm * R
        size_V = mp * R
        size_W = pn * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:]
        
        res = compute_residual(U_flat, V_flat, W_flat)
        return res.flatten()

    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    start_time = time.time()
    timeout = 110.0 # Leave 10s buffer
    
    # Try ranks from 15 to 29
    for R in range(15, 30):
        if time.time() - start_time > timeout:
            break
            
        # Try a few random initializations
        for attempt in range(20):
            if time.time() - start_time > timeout:
                break
                
            # Random initialization
            x0 = np.random.uniform(-1, 1, R * (nm + mp + pn))
            
            try:
                result = least_squares(residual, x0, args=(R,), method='lm', max_nfev=500)
                
                if result.success or result.cost < 1e-6:
                    U_flat = result.x[:nm*R]
                    V_flat = result.x[nm*R:nm*R+mp*R]
                    W_flat = result.x[nm*R+mp*R:]
                    
                    # Snap to half-integers
                    U_rounded = np.round(U_flat * 2) / 2
                    V_rounded = np.round(V_flat * 2) / 2
                    W_rounded = np.round(W_flat * 2) / 2
                    
                    # Verify rounded solution
                    res_rounded = compute_residual(U_rounded, V_rounded, W_rounded)
                    
                    if np.allclose(res_rounded, 0.0, atol=1e-6):
                        return {
                            "U": U_rounded.reshape(nm, R).tolist(),
                            "V": V_rounded.reshape(mp, R).tolist(),
                            "W": W_rounded.reshape(pn, R).tolist()
                        }
                        
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
