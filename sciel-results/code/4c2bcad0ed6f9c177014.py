# EVOLVE-BLOCK-START

def _verify(U, V, W, n=2, m=2, p=5):
    """
    Verify if the given U, V, W correctly decompose the matrix multiplication tensor.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                computed = 0.0
                for r in range(rank):
                    computed += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(computed - target) > 1e-9:
                    return False
    return True

def _construct_rank18(n=2, m=2, p=5):
    """
    Construct a rank-18 decomposition using Strassen for 2x2x2 (cols 0-1),
    Strassen for 2x2x2 (cols 2-3), and Schoolbook for 2x2x1 (col 4).
    """
    rank = 18
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
        
        r = base + 2
        U[0][r] = 1
        V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
        W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 3
        U[3][r] = 1
        V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
        W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
        
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1*p+col1][r] = 1
        W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
        
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
        W[col1*n+1][r] = 1
        
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    # Schoolbook for column 4 (4 terms)
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    return None

def _construct_schoolbook(n=2, m=2, p=5):
    """
    Construct the standard schoolbook decomposition of rank n*m*p.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _build_target_tensor(n=2, m=2, p=5):
    """Build the target tensor T[u,v,w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0
    return T

def _get_rank18_np(n=2, m=2, p=5):
    """Get rank-18 decomposition as numpy arrays."""
    rank = 18
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1; W[col1*n+1, r] = 1
        
        r = base + 1
        U[2, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1
        W[col0*n+1, r] = 1; W[col1*n+1, r] = -1
        
        r = base + 2
        U[0, r] = 1
        V[0*p+col1, r] = 1; V[1*p+col1, r] = -1
        W[col1*n+0, r] = 1; W[col1*n+1, r] = 1
        
        r = base + 3
        U[3, r] = 1
        V[1*p+col0, r] = 1; V[0*p+col0, r] = -1
        W[col0*n+0, r] = 1; W[col0*n+1, r] = 1
        
        r = base + 4
        U[0, r] = 1; U[1, r] = 1
        V[1*p+col1, r] = 1
        W[col0*n+0, r] = -1; W[col1*n+0, r] = 1
        
        r = base + 5
        U[2, r] = 1; U[0, r] = -1
        V[0*p+col0, r] = 1; V[0*p+col1, r] = 1
        W[col1*n+1, r] = 1
        
        r = base + 6
        U[1, r] = 1; U[3, r] = -1
        V[1*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    # Schoolbook for column 4
    r = 14
    U[0, r] = 1; V[0*p+4, r] = 1; W[4*n+0, r] = 1
    r = 15
    U[1, r] = 1; V[1*p+4, r] = 1; W[4*n+0, r] = 1
    r = 16
    U[2, r] = 1; V[0*p+4, r] = 1; W[4*n+1, r] = 1
    r = 17
    U[3, r] = 1; V[1*p+4, r] = 1; W[4*n+1, r] = 1
    
    return U, V, W

def _als_refine(U, V, W, T, R, n, m, p, max_iter=300, tol=1e-14, sparse_threshold=0.0):
    """Run ALS iterations to refine the decomposition."""
    prev_err = np.inf
    for it in range(max_iter):
        # Solve for U
        VvW = np.einsum('vr,wr->vwr', V, W)
        VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
        for u in range(n*m):
            b = T[u].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(VvW_flat, b, rcond=None)
            if sparse_threshold > 0:
                sol[np.abs(sol) < sparse_threshold] = 0
            U[u] = sol
        
        # Solve for V
        Uw = np.einsum('ur,wr->uwr', U, W)
        Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
        for v in range(m*p):
            b = T[:, v, :].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(Uw_flat, b, rcond=None)
            if sparse_threshold > 0:
                sol[np.abs(sol) < sparse_threshold] = 0
            V[v] = sol
        
        # Solve for W
        UV = np.einsum('ur,vr->uvr', U, V)
        UV_flat = UV.reshape(-1, R)
        for w in range(p*n):
            b = T[:, :, w].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(UV_flat, b, rcond=None)
            if sparse_threshold > 0:
                sol[np.abs(sol) < sparse_threshold] = 0
            W[w] = sol
        
        # Compute error
        err = 0.0
        for r in range(R):
            err += np.sum((np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r]) - T)**2)
        err = np.sqrt(err / (n*m * m*p * p*n))
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return U, V, W, err

def _bfgs_polish(U, V, W, T, R, n, m, p, max_iter=50):
    """Polish the solution using BFGS on the squared error."""
    def error_func(params):
        U_flat = params[:n*m*R].reshape(n*m, R)
        V_flat = params[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
        W_flat = params[n*m*R + m*p*R :].reshape(p*n, R)
        
        res = np.zeros_like(T)
        for r in range(R):
            res += np.einsum('u,v,w->uvw', U_flat[:,r], V_flat[:,r], W_flat[:,r])
        return np.sum((res - T)**2)
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    # Use scipy optimize if available, else simple gradient descent? 
    # scipy is available.
    try:
        from scipy.optimize import minimize
        res = minimize(error_func, x0, method='BFGS', options={'maxiter': max_iter, 'gtol': 1e-12})
        if res.success or res.fun < error_func(x0):
            x_opt = res.x
            U_new = x_opt[:n*m*R].reshape(n*m, R)
            V_new = x_opt[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
            W_new = x_opt[n*m*R + m*p*R :].reshape(p*n, R)
            return U_new, V_new, W_new
    except ImportError:
        pass
    
    return U, V, W

def _check_and_rationalize(U, V, W, T, R, n, m, p):
    """Check if decomposition is exact and try to rationalize entries."""
    result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
    
    if not np.allclose(result, T, atol=1e-9):
        return None
    
    # Try to rationalize with various denominators
    for denom in [2, 4, 8, 16, 32]:
        U_s = np.round(U * denom) / denom
        V_s = np.round(V * denom) / denom
        W_s = np.round(W * denom) / denom
        
        result_s = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(R):
            result_s += np.einsum('u,v,w->uvw', U_s[:,r], V_s[:,r], W_s[:,r])
        
        if np.allclose(result_s, T, atol=1e-9):
            U_list = [[float(x) for x in row] for row in U_s]
            V_list = [[float(x) for x in row] for row in V_s]
            W_list = [[float(x) for x in row] for row in W_s]
            
            for mat in [U_list, V_list, W_list]:
                for i in range(len(mat)):
                    for j in range(len(mat[i])):
                        if abs(mat[i][j] - round(mat[i][j])) < 1e-10:
                            mat[i][j] = int(round(mat[i][j]))
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return as floats if valid
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _exact_fraction_check(U_list, V_list, W_list, n, m, p):
    """Exact check using Fraction to avoid floating point issues."""
    R = len(U_list[0])
    # Convert to Fractions
    U_frac = [[Fraction(0) for _ in range(R)] for _ in range(len(U_list))]
    V_frac = [[Fraction(0) for _ in range(R)] for _ in range(len(V_list))]
    W_frac = [[Fraction(0) for _ in range(R)] for _ in range(len(W_list))]
    
    for i in range(len(U_list)):
        for j in range(R):
            val = U_list[i][j]
            nearest = round(val * 2) / 2.0
            if abs(val - nearest) < 1e-9:
                U_frac[i][j] = Fraction(int(nearest * 2), 2)
            else:
                U_frac[i][j] = Fraction(val).limit_denominator(100)
                
    for i in range(len(V_list)):
        for j in range(R):
            val = V_list[i][j]
            nearest = round(val * 2) / 2.0
            if abs(val - nearest) < 1e-9:
                V_frac[i][j] = Fraction(int(nearest * 2), 2)
            else:
                V_frac[i][j] = Fraction(val).limit_denominator(100)
                
    for i in range(len(W_list)):
        for j in range(R):
            val = W_list[i][j]
            nearest = round(val * 2) / 2.0
            if abs(val - nearest) < 1e-9:
                W_frac[i][j] = Fraction(int(nearest * 2), 2)
            else:
                W_frac[i][j] = Fraction(val).limit_denominator(100)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                total = Fraction(0)
                for r in range(R):
                    total += U_frac[u][r] * V_frac[v][r] * W_frac[w][r]
                if total != Fraction(1):
                    return False
    return True

def _search_rank(R, n, m, p, timeout=60, n_trials=30, use_structured=True):
    """
    Search for a rank-R decomposition using ALS with structured initializations.
    """
    start_time = time.time()
    
    T = _build_target_tensor(n, m, p)
    
    for trial in range(n_trials):
        if time.time() - start_time > timeout - 5:
            break
        
        if trial < 10:
            # Random initialization
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
            sparse_th = 0.0
        elif trial < 20 and use_structured:
            # Structured: start from rank-18 and drop columns
            U18, V18, W18 = _get_rank18_np(n, m, p)
            if R <= 18:
                U = U18[:, :R] + np.random.randn(n*m, R) * 0.1
                V = V18[:, :R] + np.random.randn(m*p, R) * 0.1
                W = W18[:, :R] + np.random.randn(p*n, R) * 0.1
            else:
                U = np.random.randn(n*m, R) * 0.5
                V = np.random.randn(m*p, R) * 0.5
                W = np.random.randn(p*n, R) * 0.5
            sparse_th = 0.0
        else:
            # Sparse random initialization
            scale = 0.5
            U = np.random.randn(n*m, R) * scale
            V = np.random.randn(m*p, R) * scale
            W = np.random.randn(p*n, R) * scale
            # Apply sparsity
            U[np.abs(U) < 0.3] = 0
            V[np.abs(V) < 0.3] = 0
            W[np.abs(W) < 0.3] = 0
            sparse_th = 0.1 + 0.1 * (trial - 20) / max(1, n_trials - 20)
        
        U, V, W, err = _als_refine(U, V, W, T, R, n, m, p, max_iter=100, sparse_threshold=sparse_th)
        
        # Polish with BFGS
        U, V, W = _bfgs_polish(U, V, W, T, R, n, m, p, max_iter=20)
        
        result = _check_and_rationalize(U, V, W, T, R, n, m, p)
        if result is not None:
            # Double-check with exact fraction arithmetic
            if _exact_fraction_check(result['U'], result['V'], result['W'], n, m, p):
                return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT
    
    # First, get a guaranteed constructive solution (rank 18) as fallback
    fallback = _construct_rank18(n, m, p)
    
    # Use remaining time for ALS search on lower ranks
    # Try rank 17 search
    remaining = deadline - time.time() - 5
    if remaining > 25:
        result = _search_rank(17, n, m, p, timeout=min(30, remaining), n_trials=25)
        if result is not None:
            return result
    
    # Try rank 16 search
    remaining = deadline - time.time() - 5
    if remaining > 20:
        result = _search_rank(16, n, m, p, timeout=min(25, remaining), n_trials=20)
        if result is not None:
            return result
    
    # Try rank 15 search
    remaining = deadline - time.time() - 5
    if remaining > 15:
        result = _search_rank(15, n, m, p, timeout=min(20, remaining), n_trials=15)
        if result is not None:
            return result
    
    # Return the constructive rank-18 solution
    if fallback is not None:
        return fallback
    
    # Fallback: Schoolbook (rank 20)
    return _construct_schoolbook(n, m, p)

# EVOLVE-BLOCK-END
