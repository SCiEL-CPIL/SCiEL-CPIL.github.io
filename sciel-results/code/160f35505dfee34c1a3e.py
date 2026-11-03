"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank(n, m, p, rank, seed=0, max_iters=200, deadline=None):
    """Try to find a rank-r decomposition using alternating minimization."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n*m, rank)
    V = rng.randn(m*p, rank)
    W = rng.randn(p*n, rank)
    
    best_err = float('inf')
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 2.0:
            break
        # Alternate: fix V,W, solve for U; fix U,W, solve for V; fix U,V, solve for W
        for _ in range(3):
            # Solve for U: for each (i,j), U[i*m+j,:] should satisfy
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
            # This is a linear system: U[i*m+j,:] @ (V[j*p+0,:]*W[0*n+i,:]) + ... 
            # Actually: sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1
            # Let M_k = V[j*p+k,:] * W[k*n+i,:] (element-wise)
            # Then sum_r U[i*m+j,r] * M_k[r] = 1 for each k
            # So U[i*m+j,:] @ M = 1 where M is p x rank matrix with M[k,r] = V[j*p+k,r]*W[k*n+i,r]
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve U[i*m+j,:] @ M = 1
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Solve for V: for each (j,k), V[j*p+k,:] @ (U[i*m+j,:]*W[k*n+i,:]) = 1 for all i
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, rank))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Solve for W: for each (k,i), W[k*n+i,:] @ (U[i*m+j,:]*V[j*p+k,:]) = 1 for all j
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
        
        # Check error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (s - 1.0)**2
        if err < best_err:
            best_err = err
        if best_err < 1e-6:
            break
    
    if best_err < 1e-4:
        return U, V, W, best_err
    return None, None, None, best_err

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    school_rank = n * m * p
    
    # First, get the schoolbook solution as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, school_rank
    
    # Try ranks from known lower bounds up to schoolbook rank - 1
    # For 3x3 x 3x6, known lower bounds: at least max(n*m, m*p, p*n) = max(9,18,18) = 18
    # Try a range of ranks
    ranks_to_try = list(range(18, school_rank))
    
    # Sort by rank (lower is better)
    for rank in ranks_to_try:
        if time.time() > deadline - 5.0:
            break
        found = False
        for seed in range(10):
            if time.time() > deadline - 5.0:
                break
            U, V, W, err = _try_rank(n, m, p, rank, seed=seed, max_iters=100, deadline=deadline)
            if U is not None and err < 1e-4:
                # Verify with tighter tolerance
                if _verify(U, V, W, n, m, p, tol=1e-6):
                    best_U, best_V, best_W, best_rank = U, V, W, rank
                    found = True
                    break
        if found:
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
