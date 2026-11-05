import numpy as np
import time

def construct():
    n, m, p = 3, 3, 6
    nm = n * m      # 9
    mp = m * p      # 18
    pn = p * n      # 18
    
    start_time = time.time()
    deadline = start_time + 120
    
    def _verify(U, V, W, tol=1e-9):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def _schoolbook():
        """Return the trivial rank-54 decomposition."""
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    def _als_step(U, V, W, rank):
        """One ALS iteration. Returns (U, V, W)."""
        # Update U
        for i in range(n):
            for j in range(m):
                a = i * m + j
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    U[a, :], _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                b = j * p + k
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    V[b, :], _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                c_idx = k * n + i
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    W[c_idx, :], _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                except:
                    pass
        
        return U, V, W
    
    def _error(U, V, W):
        """Compute the sum of squared errors."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0)**2
        return err
    
    best_result = None
    best_rank = 54
    
    # Try ranks from low to high
    for rank in range(38, 54):
        if time.time() > deadline - 10:
            break
        
        n_trials = 10
        for trial in range(n_trials):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(hash((rank, trial)) % (2**32))
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            U /= np.sqrt(nm)
            V /= np.sqrt(mp)
            W /= np.sqrt(pn)
            
            converged = False
            max_iters = 100
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                U, V, W = _als_step(U, V, W, rank)
                
                if it % 10 == 0 or it == max_iters - 1:
                    err = _error(U, V, W)
                    if err < 1e-12:
                        converged = True
                        break
            
            if converged:
                U_clean = np.round(U * 2) / 2
                V_clean = np.round(V * 2) / 2
                W_clean = np.round(W * 2) / 2
                
                if _verify(U_clean, V_clean, W_clean, 1e-9):
                    best_result = (U_clean, V_clean, W_clean)
                    best_rank = rank
                    break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        U, V, W = best_result
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }
    else:
        U, V, W = _schoolbook()
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }
