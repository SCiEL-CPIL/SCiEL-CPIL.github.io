"""Rewritten program for 2x2x5 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _check_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if U,V,W exactly decompose the multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _make_rank1_term(i, j, k, n, m, p):
    """Create a single rank-1 term for the (i,j,k) entry of the multiplication tensor."""
    u = np.zeros(n * m)
    v = np.zeros(m * p)
    w = np.zeros(p * n)
    u[i * m + j] = 1.0
    v[j * p + k] = 1.0
    w[k * n + i] = 1.0
    return u, v, w

def _build_from_terms(terms, n, m, p):
    """Build U, V, W from a list of (u, v, w) term tuples."""
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for r, (u, v, w) in enumerate(terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
    return U, V, W

def _schoolbook(n, m, p):
    """Return schoolbook (rank n*m*p) decomposition."""
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append(_make_rank1_term(i, j, k, n, m, p))
    return _build_from_terms(terms, n, m, p)

def _strassen_2x2x2():
    """Return 7-term Strassen decomposition for 2x2x2.
    Convention: U indexed by (i,j) with row = i*m+j, m=2
               V indexed by (j,k) with row = j*p+k, p=2
               W indexed by (k,i) with row = k*n+i, n=2
    """
    # (u_vec[4], v_vec[4], w_vec[4]) where index = row_index
    # u row: i*2+j, v row: j*2+k, w row: k*2+i
    terms = []
    # M1 = (a11+a22)(b11+b22)
    terms.append((
        np.array([1,0, 0,1]),   # a11+a22
        np.array([1,0, 0,1]),   # b11+b22
        np.array([1,0, 0,1]),   # c11 +=, c22 +=
    ))
    # M2 = (a21+a22)b11
    terms.append((
        np.array([0,0, 1,1]),   # a21+a22
        np.array([1,0, 0,0]),   # b11
        np.array([0,1, 0,1]),   # c21 +=, c22 +=
    ))
    # M3 = a11(b21-b11)
    terms.append((
        np.array([1,0, 0,0]),   # a11
        np.array([0,1, 0,-1]),  # b21-b11
        np.array([0,1, 0,1]),   # c21 +=, c11 +=
    ))
    # M4 = a22(b22-b12)
    terms.append((
        np.array([0,0, 0,1]),   # a22
        np.array([0,0, 1,-1]),  # b22-b12
        np.array([1,1, 0,0]),   # c11 +=, c12 +=
    ))
    # M5 = (a11+a12)b22
    terms.append((
        np.array([1,1, 0,0]),   # a11+a12
        np.array([0,0, 0,1]),   # b22
        np.array([1,1, 0,0]),   # c11 +=, c12 +=
    ))
    # M6 = (a21-a11)(b11+b12)
    terms.append((
        np.array([1,0, -1,0]),  # a21-a11
        np.array([1,1, 0,0]),   # b11+b12
        np.array([0,1, 0,0]),   # c21 +=
    ))
    # M7 = (a12-a22)(b21+b22)
    terms.append((
        np.array([0,1, 0,-1]),  # a12-a22
        np.array([0,0, 1,1]),   # b21+b22
        np.array([1,0, 0,1]),   # c12 +=, c22 +=
    ))
    return terms

def _extend_strassen_to_2x5(strassen_terms, n, m, p, k_offset=0):
    """Extend 2x2x2 Strassen terms to 2x2xp terms starting at column k_offset."""
    extended = []
    for u2, v2, w2 in strassen_terms:
        u_full = u2.astype(float)
        v_full = np.zeros(m * p)
        w_full = np.zeros(p * n)
        for j in range(m):
            for k in range(2):
                if k_offset + k < p:
                    v_full[j * p + (k_offset + k)] = v2[j * 2 + k]
        for k in range(2):
            if k_offset + k < p:
                for i in range(n):
                    w_full[(k_offset + k) * n + i] = w2[k * 2 + i]
        extended.append((u_full, v_full, w_full))
    return extended

def _hybrid_decomposition(n, m, p):
    """Build a hybrid decomposition: Strassen for first 2 columns + schoolbook for rest.
    Also try Strassen for other column pairs and pick the best verified one.
    """
    strassen_2x2 = _strassen_2x2x2()
    
    # Option 1: Strassen on cols 0-1, schoolbook on cols 2-4
    terms1 = _extend_strassen_to_2x5(strassen_2x2, n, m, p, k_offset=0)
    for i in range(n):
        for j in range(m):
            for k in range(2, p):
                terms1.append(_make_rank1_term(i, j, k, n, m, p))
    U1, V1, W1 = _build_from_terms(terms1, n, m, p)
    
    # Option 2: Strassen on cols 1-2, schoolbook on cols 0, 3, 4
    terms2 = _extend_strassen_to_2x5(strassen_2x2, n, m, p, k_offset=1)
    for i in range(n):
        for j in range(m):
            for k in [0, 3, 4]:
                terms2.append(_make_rank1_term(i, j, k, n, m, p))
    U2, V2, W2 = _build_from_terms(terms2, n, m, p)
    
    # Option 3: Strassen on cols 3-4, schoolbook on cols 0-2
    terms3 = _extend_strassen_to_2x5(strassen_2x2, n, m, p, k_offset=3)
    for i in range(n):
        for j in range(m):
            for k in range(0, 3):
                terms3.append(_make_rank1_term(i, j, k, n, m, p))
    U3, V3, W3 = _build_from_terms(terms3, n, m, p)
    
    # Verify and return the first valid one
    for U, V, W in [(U1, V1, W1), (U2, V2, W2), (U3, V3, W3)]:
        if _check_exact(U, V, W, n, m, p):
            return U, V, W
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)

def _tensor_squared_error(U, V, W, n, m, p):
    """Compute sum of squared errors over all (i,j,k) entries."""
    err = 0.0
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err += (s - 1.0) ** 2
    return err

def _optimize_from_start(U0, V0, W0, n, m, p, deadline, max_iter=800):
    """Try to optimize a decomposition starting from (U0, V0, W0)."""
    from scipy.optimize import minimize
    
    rank = U0.shape[1]
    
    def obj(params):
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank + m*p*rank :].reshape(p*n, rank)
        return _tensor_squared_error(U, V, W, n, m, p)
    
    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    result = minimize(obj, x0, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-18, 'gtol': 1e-15})
    
    if result.fun < 1e-12:
        U = result.x[:n*m*rank].reshape(n*m, rank)
        V = result.x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        W = result.x[n*m*rank + m*p*rank :].reshape(p*n, rank)
        if _check_exact(U, V, W, n, m, p, tol=1e-9):
            return U, V, W
    
    return None

def _try_reduce(U, V, W, n, m, p, target_rank, deadline, n_trials=6):
    """Attempt to find a valid rank-target_rank decomposition."""
    rng = np.random.RandomState(target_rank * 31 + 7)
    current_rank = U.shape[1]
    
    if target_rank >= current_rank:
        return None
    
    # Strategy A: column sampling with perturbation
    for trial in range(n_trials):
        if time.time() > deadline - 20:
            return None
        
        # Weighted column selection based on term "importance"
        norms = np.linalg.norm(U, axis=0) * np.linalg.norm(V, axis=0) * np.linalg.norm(W, axis=0)
        weights = norms / (norms.sum() + 1e-15)
        
        cols = rng.choice(current_rank, size=target_rank, replace=False, p=weights)
        
        U_init = U[:, cols].copy() + rng.randn(n*m, target_rank) * 0.05
        V_init = V[:, cols].copy() + rng.randn(m*p, target_rank) * 0.05
        W_init = W[:, cols].copy() + rng.randn(p*n, target_rank) * 0.05
        
        res = _optimize_from_start(U_init, V_init, W_init, n, m, p, deadline, max_iter=500)
        if res is not None:
            return res
    
    # Strategy B: random initialization
    for trial in range(3):
        if time.time() > deadline - 10:
            return None
        
        U_init = rng.randn(n*m, target_rank) * 0.3
        V_init = rng.randn(m*p, target_rank) * 0.3
        W_init = rng.randn(p*n, target_rank) * 0.3
        
        res = _optimize_from_start(U_init, V_init, W_init, n, m, p, deadline, max_iter=400)
        if res is not None:
            return res
    
    # Strategy C: no-perturbation column subset
    for trial in range(2):
        if time.time() > deadline - 5:
            return None
        
        cols = rng.choice(current_rank, size=target_rank, replace=False)
        U_init = U[:, cols].copy()
        V_init = V[:, cols].copy()
        W_init = W[:, cols].copy()
        
        res = _optimize_from_start(U_init, V_init, W_init, n, m, p, deadline, max_iter=600)
        if res is not None:
            return res
    
    return None

def _snap_values(mat, tol=1e-10):
    """Snap near-integer and near-half-integer values to exact ones."""
    mat = mat.copy()
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if abs(v - round(v)) < tol:
                mat[i, j] = round(v)
            elif abs(2*v - round(2*v)) < tol:
                mat[i, j] = round(2*v) / 2.0
    return mat

def construct():
    start = time.time()
    deadline = start + 120  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 2, 5
    
    # Build initial decomposition
    U, V, W = _hybrid_decomposition(n, m, p)
    
    if not _check_exact(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        if not _check_exact(U, V, W, n, m, p):
            # Should never happen, but return something
            rank = n * m * p
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, c] = 1
                        V[j*p+k, c] = 1
                        W[k*n+i, c] = 1
                        c += 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank: attempt each target rank from best_rank-1 down to max(10, ...)
    # Lower bound: for 2x2x5, rank >= max(n*m, m*p, p*n) = max(4,10,10) = 10
    # Also rank >= n*m*p / max(n,m,p) by a simple bound, but that's not tight.
    # Realistic lower bound for 2x2x5 is around 10-12.
    
    lower_bound = 10  # conservative
    
    for target in range(best_rank - 1, lower_bound - 1, -1):
        if time.time() > deadline - 30:
            break
        
        res = _try_reduce(best_U, best_V, best_W, n, m, p, target, deadline, n_trials=4)
        if res is not None:
            U_r, V_r, W_r = res
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = target
            # Continue trying to go even lower
            # Don't break; keep trying
    
    # Final polish on the best solution found
    if time.time() < deadline - 8:
        from scipy.optimize import minimize
        rank = best_U.shape[1]
        
        def obj(params):
            U = params[:n*m*rank].reshape(n*m, rank)
            V = params[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
            W = params[n*m*rank + m*p*rank :].reshape(p*n, rank)
            return _tensor_squared_error(U, V, W, n, m, p)
        
        x0 = np.concatenate([best_U.flatten(), best_V.flatten(), best_W.flatten()])
        result = minimize(obj, x0, method='L-BFGS-B',
                          options={'maxiter': 2000, 'ftol': 1e-20, 'gtol': 1e-16})
        
        if result.fun < 1e-14:
            U_p = result.x[:n*m*rank].reshape(n*m, rank)
            V_p = result.x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
            W_p = result.x[n*m*rank + m*p*rank :].reshape(p*n, rank)
            if _check_exact(U_p, V_p, W_p, n, m, p, tol=1e-10):
                best_U, best_V, best_W = U_p, V_p, W_p
    
    # Snap to clean values
    best_U = _snap_values(best_U)
    best_V = _snap_values(best_V)
    best_W = _snap_values(best_W)
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
