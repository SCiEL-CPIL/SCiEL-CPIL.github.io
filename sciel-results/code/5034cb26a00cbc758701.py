<<<<<<< SEARCH
def _try_rank(r, time_limit=5, seed=0):
    """Try to find a rank-r decomposition. Returns (U,V,W) or None."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Reduced attempts to prevent timeout
    start = time.time()
    max_attempts = 200
    
    for attempt in range(max_attempts):
        if time.time() - start > time_limit:
            break
        
        # Random initialization with small values
        U = np_rng.randn(n*m, r) * 0.5
        V = np_rng.randn(m*p, r) * 0.5
        W = np_rng.randn(p*n, r) * 0.5
        
        # Iterative refinement: update one factor at a time
        for iteration in range(50):
            i = rng.randint(0, n-1)
            j = rng.randint(0, m-1)
            k = rng.randint(0, p-1)
            
            a = i * m + j
            b = j * p + k
            c_idx = k * n + i
            
            # Current value
            cur = np.dot(U[a], V[b] * W[c_idx])
            err = 1.0 - cur
            
            lr = 0.1
            U[a] += lr * err * V[b] * W[c_idx]
            V[b] += lr * err * U[a] * W[c_idx]
            W[c_idx] += lr * err * U[a] * V[b]
        
        # Check error
        err_sum = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c_idx = k * n + i
                    val = np.dot(U[a], V[b] * W[c_idx])
                    err_sum += (val - 1)**2
        
        if err_sum < best_err:
            best_err = err_sum
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
            if best_err < 1e-12:
                return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None
=======
def _try_rank(r, time_limit=5, seed=0):
    """Try to find a rank-r decomposition using ALS. Returns (U,V,W) or None."""
    np_rng = np.random.RandomState(seed)
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    start = time.time()
    max_attempts = 50
    
    for attempt in range(max_attempts):
        if time.time() - start > time_limit:
            break
        
        # Random initialization
        U = np_rng.randn(n*m, r)
        V = np_rng.randn(m*p, r)
        W = np_rng.randn(p*n, r)
        
        # Alternating Least Squares: fix two factors, solve for the third
        for iteration in range(100):
            if time.time() - start > time_limit:
                break
            
            # Update U: for each row a = i*m+j, solve U[a] @ (V[b]*W[c]) = 1
            # This is a least squares problem
            for i in range(n):
                for j in range(m):
                    a = i * m + j
                    # Build system: sum_k U[a,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                    # Actually we need: for each k, sum_c U[a,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                    # This couples different k's, so we need to solve jointly
                    # Better approach: treat as linear system in U[a]
                    # For fixed j, the constraint is: for all k, sum_c U[a,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
                    # Let M be matrix where M[k,c] = V[j*p+k,c] * W[k*n+i,c]
                    # Then M @ U[a] = ones(p)
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve M @ x = ones(p)
                    try:
                        U[a, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each row b = j*p+k, solve V[b] @ (U[a]*W[c]) = 1
            for j in range(m):
                for k in range(p):
                    b = j * p + k
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[b, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each row c = k*n+i, solve W[c] @ (U[a]*V[b]) = 1
            for k in range(p):
                for i in range(n):
                    c_idx = k * n + i
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[c_idx, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        pass
        
        # Check error
        err_sum = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c_idx = k * n + i
                    val = np.dot(U[a], V[b] * W[c_idx])
                    err_sum += (val - 1)**2
        
        if err_sum < best_err:
            best_err = err_sum
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
            if best_err < 1e-12:
                return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None
>>>>>>> REPLACE

