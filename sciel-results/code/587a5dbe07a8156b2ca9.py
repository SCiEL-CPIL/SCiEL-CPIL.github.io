import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Indices
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    
    def error_fn(params, R):
        """
        params: flattened array of size 3*R*nm? No, U is (nm, R), V is (mp, R), W is (pn, R).
        Total size: nm*R + mp*R + pn*R = R*(nm+mp+pn).
        """
        # Reshape
        size_U = nm * R
        size_V = mp * R
        size_W = pn * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:size_U+size_V+size_W]
        
        U = U_flat.reshape(nm, R)
        V = V_flat.reshape(mp, R)
        W = W_flat.reshape(pn, R)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    val = np.dot(U[a], V[b]) # Wait, sum over r of U[a,r] V[b,r] W[c,r]
                    # np.dot(U[a], V[b]) is sum U[a,r] V[b,r].
                    # We need sum U[a,r] V[b,r] W[c,r].
                    # This is element-wise product of vectors, then dot?
                    # Let X = U[a] * V[b]. Then val = np.dot(X, W[c]).
                    val = np.dot(U[a] * V[b], W[c])
                    err += (val - 1.0)**2
        return err

    def get_solution(params, R):
        size_U = nm * R
        size_V = mp * R
        size_W = pn * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:size_U+size_V+size_W]
        
        U = U_flat.reshape(nm, R)
        V = V_flat.reshape(mp, R)
        W = W_flat.reshape(pn, R)
        
        return U, V, W

    def check_and_return(U, V, W, tol=1e-6):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    val = np.dot(U[a] * V[b], W[c])
                    if abs(val - 1.0) > tol:
                        return None
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def snap_to_grid(arr):
        # Snap to nearest 0.5
        return np.round(arr * 2) / 2

    best_result = None
    best_rank = 24
    
    # Naive solution as fallback
    U_naive = np.zeros((nm, 24))
    V_naive = np.zeros((mp, 24))
    W_naive = np.zeros((pn, 24))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i*m*p + j*p + k
                U_naive[i*m+j, r] = 1
                V_naive[j*p+k, r] = 1
                W_naive[k*n+i, r] = 1
    best_result = check_and_return(U_naive, V_naive, W_naive)
    
    # Try to find lower rank
    # Known rank for 2x3x4 is 19.
    # Try ranks 19, 20, 21, 22, 23
    for R in range(19, 24):
        if time.time() > deadline - 30:
            break
            
        found = False
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(attempt * 100 + R)
            # Initialize with small random values
            size_U = nm * R
            size_V = mp * R
            size_W = pn * R
            total_size = size_U + size_V + size_W
            
            x0 = np.random.randn(total_size) * 0.5
            
            res = minimize(error_fn, x0, args=(R,), method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
            
            if res.success or res.fun < 1e-10:
                U, V, W = get_solution(res.x, R)
                
                # Try to snap to grid
                U_snapped = snap_to_grid(U)
                V_snapped = snap_to_grid(V)
                W_snapped = snap_to_grid(W)
                
                # Check snapped
                res_snapped = check_and_return(U_snapped, V_snapped, W_snapped, tol=1e-9)
                if res_snapped is not None:
                    best_rank = R
                    best_result = res_snapped
                    found = True
                    break
                
                # Check original if snapped failed (maybe it's a valid float solution but not snapped? 
                # Problem requires integer/half-integer. So snapped is mandatory for validity?)
                # "integer/half-integer entries that exactly decompose"
                # So we MUST return snapped values.
                
        if found:
            break

    if best_result is None:
        # Fallback to naive
        best_result = {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}

    return best_result
