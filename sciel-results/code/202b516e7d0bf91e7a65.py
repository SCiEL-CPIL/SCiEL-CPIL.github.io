import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _validate(U, V, W):
    """
    Validate that U, V, W form an exact decomposition of the matmul(3,5,6) tensor.
    """
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    r = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if not all(len(row) == r for row in U):
        return False
    if not all(len(row) == r for row in V):
        return False
    if not all(len(row) == r for row in W):
        return False
    if r == 0:
        return False
    
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m + j] * V_np[j*p + k] * W_np[k*n + i])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _compute_error(U, V, W, n, m, p):
    """Compute squared error of decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m + j], V[j*p + k] * W[k*n + i])
                err += (val - 1.0) ** 2
    return np.sqrt(err)

def _alternating_min(U, V, W, n, m, p, max_iters=200, tol=1e-12, deadline=None):
    """
    Alternating least squares to refine the decomposition.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Update U: for each (i,j), solve for U[i*m+j] given V,W
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p + k] * W[k*n + i]
                try:
                    U[i*m + j] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V: for each (j,k), solve for V[j*p+k] given U,W
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m + j] * W[k*n + i]
                try:
                    V[j*p + k] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W: for each (k,i), solve for W[k*n+i] given U,V
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i*m + j] * V[j*p + k]
                try:
                    W[k*n + i] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        error = _compute_error(U, V, W, n, m, p)
        if error < tol:
            break
    
    error = _compute_error(U, V, W, n, m, p)
    if error < tol:
        return U, V, W, error
    return None, None, None, error

def _round_to_half_integers(arr):
    """Round array to nearest half-integers."""
    return np.round(arr * 2) / 2

def _try_rank(rank, n, m, p, deadline, seed_base=0):
    """
    Try to find a valid decomposition of given rank using multiple initialization strategies.
    """
    max_attempts = 15
    tuples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 10:
            break
        
        seed = seed_base + attempt * 7919 + rank * 104729
        np.random.seed(seed)
        
        # Strategy 1: Structured initialization (attempts 0-3)
        if attempt < 4:
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            for c in range(rank):
                idx = c % len(tuples)
                i, j, k = tuples[idx]
                U[i*m + j, c] = 1.0 + 0.05 * np.random.randn()
                V[j*p + k, c] = 1.0 + 0.05 * np.random.randn()
                W[k*n + i, c] = 1.0 + 0.05 * np.random.randn()
                
                # Add cross-terms for coverage
                if c % 2 == 0:
                    i2 = (i + 1) % n
                    j2 = (j + 1) % m
                    U[i2*m + j, c] = 0.1 * np.random.randn()
                    V[j2*p + k, c] = 0.1 * np.random.randn()
        
        # Strategy 2: Random with scaling (attempts 4-9)
        elif attempt < 10:
            U = np.random.randn(n * m, rank) * 0.3
            V = np.random.randn(m * p, rank) * 0.3
            W = np.random.randn(p * n, rank) * 0.3
        
        # Strategy 3: Sparse (attempts 10-14)
        else:
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            for c in range(rank):
                idx_u = np.random.randint(n * m)
                idx_v = np.random.randint(m * p)
                idx_w = np.random.randint(p * n)
                U[idx_u, c] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
                V[idx_v, c] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
                W[idx_w, c] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
        
        # Run alternating minimization
        U2, V2, W2, err = _alternating_min(U, V, W, n, m, p, max_iters=100, tol=1e-12, deadline=deadline)
        if U2 is not None:
            U, V, W = U2, V2, W2
        
        # Round to half-integers and validate
        U_rounded = _round_to_half_integers(U)
        V_rounded = _round_to_half_integers(V)
        W_rounded = _round_to_half_integers(W)
        
        U_list = U_rounded.tolist()
        V_list = V_rounded.tolist()
        W_list = W_rounded.tolist()
        
        if _validate(U_list, V_list, W_list):
            return U_list, V_list, W_list
    
    return None

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing lists of lists.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j][col] = 1
                V_sb[j*p + k][col] = 1
                W_sb[k*n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = [row[:] for row in U_sb], [row[:] for row in V_sb], [row[:] for row in W_sb]
    best_rank = schoolbook_rank
    
    # Lower bound
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower_bound upward
    for rank in range(lower_bound, min(lower_bound + 40, schoolbook_rank)):
        if time.time() > deadline - 15:
            break
        
        result = _try_rank(rank, n, m, p, deadline)
        
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = rank
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
