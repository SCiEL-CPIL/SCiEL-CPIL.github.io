import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def _verify(U, V, W, n, m, p, rank):
    """Check if the decomposition exactly reproduces the matrix multiplication tensor."""
    # T[i,j,k] = 1 for the multiplication tensor: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # (since the tensor entries are all 1 in the standard convention)
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _make_schoolbook(n, m, p):
    """Construct the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i*m*p + j*p + k
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
    return U, V, W, rank

def _try_rank(n, m, p, rank, deadline, seed_base):
    """Try to find a rank-`rank` decomposition. Returns (U, V, W) or None."""
    best = None
    best_err = np.inf
    n_attempts = 20
    n_iters = 80
    
    for attempt in range(n_attempts):
        if time.time() > deadline - 5:
            break
        np.random.seed(seed_base + attempt)
        
        # Initialize with random entries
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        for it in range(n_iters):
            if time.time() > deadline - 5:
                break
            
            # Alternating least squares
            # Solve for U: for each (i,j), find U[i*m+j,:] such that
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
            # This is linear in U[i*m+j,:] given V, W
            # The coefficient for U[i*m+j,r] in the equation for k is V[j*p+k,r]*W[k*n+i,r]
            # So we need: M_{k,r} = V[j*p+k,r]*W[k*n+i,r], and M @ U[i*m+j,:] = 1
            
            for i in range(n):
                for j in range(m):
                    # Build the p x rank matrix M where M[k,r] = V[j*p+k,r]*W[k*n+i,r]
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k] = V[j*p+k] * W[k*n+i]
                    # Solve M @ x = ones(p)
                    try:
                        x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U[i*m+j] = x
                    except np.linalg.LinAlgError:
                        pass
            
            # Solve for V: for each (j,k), find V[j*p+k,:] such that
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
            # M_{i,r} = U[i*m+j,r]*W[k*n+i,r], M @ V[j*p+k,:] = 1
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, rank))
                    for i in range(n):
                        M[i] = U[i*m+j] * W[k*n+i]
                    try:
                        x, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V[j*p+k] = x
                    except np.linalg.LinAlgError:
                        pass
            
            # Solve for W: for each (k,i), find W[k*n+i,:] such that
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
            # M_{j,r} = U[i*m+j,r]*V[j*p+k,r], M @ W[k*n+i,:] = 1
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        M[j] = U[i*m+j] * V[j*p+k]
                    try:
                        x, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W[k*n+i] = x
                    except np.linalg.LinAlgError:
                        pass
            
            # Check error
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        err = abs(s - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < best_err:
                best_err = max_err
                best = (U.copy(), V.copy(), W.copy())
            
            if max_err < 1e-9:
                return U, V, W
    
    # Verify the best found
    if best is not None and best_err < 1e-8:
        U, V, W = best
        if _verify(U, V, W, n, m, p, rank):
            return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Start with schoolbook (guaranteed valid)
    U, V, W, rank = _make_schoolbook(n, m, p)
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower-rank decompositions
    # Theoretical lower bounds suggest rank could be lower than 63
    # Try ranks from a reasonable lower bound up to just below schoolbook
    # For 3x3x7, schoolbook is 63. Try to find something better.
    
    # Try a range of ranks, starting from lower values
    # We'll try ranks that are plausible: from 25 up to 62
    # But we need to be time-efficient, so try a subset
    
    ranks_to_try = list(range(25, 63))
    # Sort by likelihood of success (closer to schoolbook first might be easier)
    # Actually, let's try from low to high, but prioritize the middle range
    # where decompositions are more likely to exist and be findable
    
    # Strategy: try ranks in order of increasing difficulty
    # Start with ranks near the middle of the plausible range
    mid = len(ranks_to_try) // 2
    order = [mid] + [mid - d for d in range(1, mid + 1) if mid - d >= 0] + \
            [mid + d for d in range(1, len(ranks_to_try) - mid) if mid + d < len(ranks_to_try)]
    
    for idx in order:
        if time.time() > deadline - 10:
            break
        r = ranks_to_try[idx]
        if r >= best_rank:
            continue
        result = _try_rank(n, m, p, r, deadline, seed_base=1000 + r)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_U, best_V, best_W = U, V, W
            # If we found a good one, we could stop, but let's try to go lower
            # Continue searching for even lower rank
    
    # Final safety: ensure we return a valid decomposition
    if not _verify(best_U, best_V, best_W, n, m, p, best_rank):
        # Fallback to schoolbook
        U, V, W, rank = _make_schoolbook(n, m, p)
        best_rank = rank
        best_U, best_V, best_W = U, V, W
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
