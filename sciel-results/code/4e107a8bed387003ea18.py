"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _tensor_error(U, V, W, n, m, p):
    """
    Compute the decomposition error for all (i,j,k) triples.
    Returns a 3D array err[i,j,k] = |sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1|
    Also returns the number of entries with error > 1e-9.
    """
    rank = U.shape[1]
    # Build the full tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Vectorized: for each r, outer product contribution
    # T = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]
    
    # Reshape U: (n*m, rank) -> (n, m, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Use einsum
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    err = np.abs(T - 1.0)
    n_bad = int(np.sum(err > 1e-9))
    return T, err, n_bad


def _random_half_integer(n, scale=3):
    """Generate a random array with half-integer entries in [-scale, scale]."""
    # Half-integers: k/2 for integer k
    nvals = 2 * scale + 1  # from -scale to scale in steps of 0.5 -> 4*scale+1 values
    # Actually: -3, -2.5, -2, ..., 2, 2.5, 3 -> 13 values for scale=3
    vals = np.arange(-scale, scale + 0.5, 0.5)
    return vals[np.random.randint(0, len(vals), size=n)]


def _random_sparse_half_integer(n, density=0.3, scale=2):
    """Generate a sparse random half-integer array."""
    vals = np.arange(-scale, scale + 0.5, 0.5)
    arr = np.zeros(n)
    mask = np.random.random(n) < density
    arr[mask] = vals[np.random.randint(0, len(vals), size=int(mask.sum()))]
    return arr


def _try_random_rank(R, n, m, p, max_iters=500, deadline=None):
    """
    Try to find a valid rank-R decomposition via random search with local repair.
    Returns (U, V, W) if found, else None.
    """
    best_T = None
    best_err = None
    best_nbad = n * m * p + 1
    best_U = best_V = best_W = None
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random initialization with sparse half-integer entries
        U = _random_sparse_half_integer(n * m, density=0.4, scale=2).reshape(n * m, 1).repeat(R, axis=1)
        V = _random_sparse_half_integer(m * p, density=0.4, scale=2).reshape(m * p, 1).repeat(R, axis=1)
        W = _random_sparse_half_integer(p * n, density=0.4, scale=2).reshape(p * n, 1).repeat(R, axis=1)
        
        # Add some structure: make some columns be schoolbook-like
        for r in range(min(R, n * m * p)):
            i = r // (m * p)
            j = (r // p) % m
            k = r % p
            if random.random() < 0.3:
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
        
        # Evaluate
        T, err, nbad = _tensor_error(U, V, W, n, m, p)
        
        if nbad < best_nbad:
            best_nbad = nbad
            best_T = T.copy()
            best_err = err.copy()
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            if nbad == 0:
                return U, V, W
        
        # Local repair: try to fix the worst entries
        if nbad > 0 and nbad < best_nbad:
            # Find the worst (i,j,k)
            worst_idx = np.unravel_index(np.argmax(err), err.shape)
            wi, wj, wk = worst_idx
            # Try modifying random entries of random factors
            for _ in range(20):
                r = random.randint(0, R - 1)
                factor = random.choice(['U', 'V', 'W'])
                if factor == 'U':
                    idx = random.randint(0, n * m - 1)
                    U[idx, r] += random.choice([-0.5, 0.5])
                elif factor == 'V':
                    idx = random.randint(0, m * p - 1)
                    V[idx, r] += random.choice([-0.5, 0.5])
                else:
                    idx = random.randint(0, p * n - 1)
                    W[idx, r] += random.choice([-0.5, 0.5])
                
                T2, err2, nbad2 = _tensor_error(U, V, W, n, m, p)
                if nbad2 < nbad:
                    nbad = nbad2
                    T = T2
                    err = err2
                    if nbad < best_nbad:
                        best_nbad = nbad
                        best_T = T.copy()
                        best_err = err.copy()
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        if nbad == 0:
                            return U, V, W
                else:
                    # Revert
                    if factor == 'U':
                        U[idx, r] -= random.choice([-0.5, 0.5])
                    elif factor == 'V':
                        V[idx, r] -= random.choice([-0.5, 0.5])
                    else:
                        W[idx, r] -= random.choice([-0.5, 0.5])
    
    return None


def _structured_rank_search(n, m, p, deadline):
    """
    Structured search: start from schoolbook and try to merge terms.
    Also try various rank values from low to high.
    """
    # First, try a few specific rank values with structured initialization
    # The theoretical lower bound for matmul rank is at least max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # Known upper bounds are much lower than n*m*p = 90
    
    # Try ranks from a reasonable range
    # Use block Strassen as a starting point for the best known
    best_result = None
    best_rank = n * m * p
    
    # Try to find a decomposition at various ranks
    for R in range(50, best_rank + 1, 2):  # Try even ranks first (half-integers)
        if time.time() > deadline:
            break
        
        result = _try_random_rank(R, n, m, p, max_iters=200, deadline=deadline)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best_result = (U, V, W)
                best_rank = R
                break
    
    return best_result, best_rank


def _block_strassen(n, m, p):
    """
    Block Strassen decomposition as a baseline.
    """
    rank_terms = []
    
    def add_schoolbook(i, j, k):
        U = np.zeros(n * m)
        V = np.zeros(m * p)
        W = np.zeros(p * n)
        U[i * m + j] = 1.0
        V[j * p + k] = 1.0
        W[k * n + i] = 1.0
        rank_terms.append((U, V, W))
    
    def add_strassen_block(i0, j0, k0):
        terms_2x2 = [
            (np.array([1, 0, 0, -1.0]), np.array([0, 0, 1, 1.0]), np.array([1, 0, 0, 1.0])),
            (np.array([0, 0, 1, 1.0]), np.array([1, 0, 0, 0.0]), np.array([0, 0, 1, 0.0])),
            (np.array([1, 0, 0, 0.0]), np.array([1, 0, 0, -1.0]), np.array([0, 0, 1, 0.0])),
            (np.array([0, 0, 0, 1.0]), np.array([0, -1, 0, 1.0]), np.array([1, 0, 1, 0.0])),
            (np.array([1, 1, 0, 0.0]), np.array([0, 0, 0, 1.0]), np.array([0, -1, 0, 1.0])),
            (np.array([-1, 0, 1, 0.0]), np.array([1, 1, 0, 0.0]), np.array([0, 0, 0, 1.0])),
            (np.array([0, 1, 0, -1.0]), np.array([0, 0, 1, 1.0]), np.array([0, 1, 0, 1.0])),
        ]
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0.0
                    for (u, v, w) in terms_2x2:
                        s += u[i * 2 + j] * v[j * 2 + k] * w[k * 2 + i]
                    if abs(s - 1.0) > 1e-9:
                        for di in range(2):
                            for dj in range(2):
                                for dk in range(2):
                                    add_schoolbook(i0 + di, j0 + dj, k0 + dk)
                        return
        
        for (u, v, w) in terms_2x2:
            U = np.zeros(n * m)
            V = np.zeros(m * p)
            W = np.zeros(p * n)
            for di in range(2):
                for dj in range(2):
                    U[(i0 + di) * m + (j0 + dj)] = u[di * 2 + dj]
            for dj in range(2):
                for dk in range(2):
                    V[(j0 + dj) * p + (k0 + dk)] = v[dj * 2 + dk]
            for dk in range(2):
                for di in range(2):
                    W[(k0 + dk) * n + (i0 + di)] = w[dk * 2 + di]
            rank_terms.append((U, V, W))
    
    covered = set()
    strassen_blocks = []
    for i0 in range(0, n - 1, 2):
        for j0 in range(0, m - 1, 2):
            for k0 in range(0, p - 1, 2):
                strassen_blocks.append((i0, j0, k0))
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            covered.add((i0 + di, j0 + dj, k0 + dk))
    
    for (i0, j0, k0) in strassen_blocks:
        add_strassen_block(i0, j0, k0)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_schoolbook(i, j, k)
    
    if not rank_terms:
        return None
    
    total_rank = len(rank_terms)
    U_mat = np.zeros((n * m, total_rank))
    V_mat = np.zeros((m * p, total_rank))
    W_mat = np.zeros((p * n, total_rank))
    
    for r, (U, V, W) in enumerate(rank_terms):
        U_mat[:, r] = U
        V_mat[:, r] = V
        W_mat[:, r] = W
    
    if _verify(U_mat, V_mat, W_mat, n, m, p):
        return {"U": U_mat, "V": V_mat, "W": W_mat}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    np.random.seed(42)
    random.seed(42)
    
    # Strategy 1: Block Strassen (deterministic, known to give ~84)
    result = _block_strassen(n, m, p)
    best_rank = n * m * p
    if result is not None:
        best_rank = result["U"].shape[1]
    
    # Strategy 2: Random tensor-rank search for lower ranks
    # Try to beat the block Strassen rank
    for R in range(max(30, best_rank - 10), best_rank + 1):
        if time.time() > deadline:
            break
        if R >= best_rank:
            continue
        
        found = _try_random_rank(R, n, m, p, max_iters=300, deadline=deadline)
        if found is not None:
            U, V, W = found
            if _verify(U, V, W, n, m, p):
                best_rank = R
                result = {"U": U, "V": V, "W": W}
                break
    
    # Fallback to schoolbook if nothing found
    if result is None:
        result = _schoolbook(n, m, p)
    
    return {"U": result["U"].tolist(), "V": result["V"].tolist(), "W": result["W"].tolist()}

# EVOLVE-BLOCK-END
