import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, R

    U_fb, V_fb, W_fb, R_fb = schoolbook()
    
    best_rank = n * m * p
    best_factors = None
    
    def compute_error(U, V, W, R):
        """Compute tensor reconstruction error efficiently using vectorized ops."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    err += (val - 1.0) ** 2
        return err
    
    def polish(U, V, W, R, max_iters=500, tol=1e-12):
        """High-precision polish pass with tighter tolerance."""
        for it in range(max_iters):
            if time.time() > deadline:
                return U, V, W
            
            # Update U
            for i in range(n):
                for j in range(m):
                    M = np.array([V[j*p+k] * W[k*n+i] for k in range(p)])
                    try:
                        U[i*m+j] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.array([U[i*m+j] * W[k*n+i] for i in range(n)])
                    try:
                        V[j*p+k] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.array([U[i*m+j] * V[j*p+k] for j in range(m)])
                    try:
                        W[k*n+i] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            err = compute_error(U, V, W, R)
            if err < tol:
                break
        
        return U, V, W
    
    def try_als(R, seed, max_iters=150, tol=1e-8):
        """Try ALS with given rank and seed. Returns (U, V, W) if converged, else None."""
        np.random.seed(seed)
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        prev_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            for i in range(n):
                for j in range(m):
                    M = np.array([V[j*p+k] * W[k*n+i] for k in range(p)])
                    try:
                        U[i*m+j] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.array([U[i*m+j] * W[k*n+i] for i in range(n)])
                    try:
                        V[j*p+k] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.array([U[i*m+j] * V[j*p+k] for j in range(m)])
                    try:
                        W[k*n+i] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            err = compute_error(U, V, W, R)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < tol:
                return U, V, W
            
            if err > prev_err * 1.01:
                break
            prev_err = err
        
        # If we didn't converge but got close, try polish
        if best_err < 1e-4:
            U, V, W = polish(best_U, best_V, best_W, R, max_iters=200, tol=1e-12)
            err = compute_error(U, V, W, R)
            if err < 1e-8:
                return U, V, W
        
        return None
    
    def snap_and_verify(U, V, W, R):
        """Snap to half-integers and verify exactness. Try integers first."""
        # Try integer snap
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        err_int = compute_error(U_int, V_int, W_int, R)
        if err_int < 1e-8:
            return U_int, V_int, W_int
        
        # Try half-integer snap
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        err = compute_error(U_snap, V_snap, W_snap, R)
        if err < 1e-8:
            return U_snap, V_snap, W_snap
        return None
    
    # Try ranks from 54 upward with multiple restarts
    for R in range(54, min(63, best_rank)):
        if time.time() > deadline:
            break
        
        found = False
        # Use more attempts for lower ranks since they're more valuable
        num_attempts = 30 if R == 54 else 20
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            seed = 42 + attempt * 17 + R * 100
            result = try_als(R, seed=seed, max_iters=150)
            if result is None:
                continue
            
            U, V, W = result
            snapped = snap_and_verify(U, V, W, R)
            if snapped is not None:
                U_s, V_s, W_s = snapped
                best_rank = R
                best_factors = (U_s, V_s, W_s)
                found = True
                break
        
        if found and R == 54:
            break
    
    if best_factors:
        U, V, W = best_factors
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
