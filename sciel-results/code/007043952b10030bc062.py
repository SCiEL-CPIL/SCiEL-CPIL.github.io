import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = 2, 5, 5
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Indices for the 50 constraints
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    # U index: i*m+j
    # V index: j*p+k
    # W index: k*n+i
    
    def residual(params, R):
        # params is a flattened vector of U, V, W
        U = params[:nm*R].reshape(nm, R)
        V = params[nm*R:nm*R+mp*R].reshape(mp, R)
        W = params[nm*R+mp*R:].reshape(pn, R)
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # Vectorized:
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    res[idx] = val - 1.0
                    idx += 1
        return res

    def try_rank(R, n_restarts=50):
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(42 + restart)
            # Initialize
            # Scale matters. If values are too small, gradient is small.
            # If too large, overflow.
            # Let's try uniform distribution in [-1, 1]
            x0 = np.random.uniform(-1, 1, size=(nm + mp + pn) * R)
            
            # Run least_squares
            try:
                result = least_squares(residual, x0, args=(R,), method='trf', max_nfev=2000)
            except Exception:
                continue
            
            if result.success or result.cost < 1e-12:
                # Verify
                U = result.x[:nm*R].reshape(nm, R)
                V = result.x[nm*R:nm*R+mp*R].reshape(mp, R)
                W = result.x[nm*R+mp*R:].reshape(pn, R)
                
                # Check max error
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                            
                if max_err < 1e-8:
                    return U.tolist(), V.tolist(), W.tolist()
            
            # Also check if we got close but not quite, maybe a polish pass?
            # For now, just return None if not successful
    
    # Schoolbook
    def build_schoolbook():
        R = n * m * p
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U.tolist(), V.tolist(), W.tolist()
    
    best_solution = build_schoolbook()
    best_rank = n * m * p
    
    # Try ranks from 50 down to 40
    for R in range(best_rank - 1, 39, -1):
        if time.time() > deadline - 10:
            break
        
        # Increase restarts for lower ranks? Or keep constant?
        # Lower ranks are harder, so maybe more restarts.
        n_restarts = 100 if R < 45 else 50
        
        result = try_rank(R, n_restarts)
        if result is not None:
            best_rank = R
            best_solution = result
            # If we found a lower rank, we can continue to see if we can go even lower
            # But if we found 46, maybe 45 is possible?
            continue
        else:
            # If we failed at R, maybe R-1 is also impossible?
            # But it's not guaranteed.
            # However, if we fail at 49, 48, 47, 46... it might be that 50 is minimal.
            # Or we just need more restarts.
            pass
    
    return {"U": best_solution[0], "V": best_solution[1], "W": best_solution[2]}
