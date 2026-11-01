"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]
import numpy as np
import time
import itertools

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W give an exact decomposition of the matmul tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Build the target tensor: T[i,j,k] = 1 for all valid (i,j,k)
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # For each column c, the contribution is U[:,c] * V[:,c] * W[:,c]
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should sum to 1 for each (i,j,k)
    total = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total[i, j, k] += uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
    # Check exactness
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(total[i,j,k] - 1.0) > 1e-9:
                    return False
    return True

def _is_half_integer(x):
    """Check if x is integer or half-integer."""
    return abs(x * 2 - round(x * 2)) < 1e-9

def _check_half_integer_matrices(U, V, W):
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not _is_half_integer(val):
                    return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = n * m * p
    
    # Try to find lower rank decompositions using optimization
    # Strategy: for each candidate rank r, try to find U, V, W with half-integer entries
    # that exactly decompose the tensor.
    
    # Use a randomized local search: start from schoolbook, try to eliminate columns
    # by finding linear dependencies or using optimization.
    
    # Alternative: direct search over small half-integer values
    # For rank r, each of U, V, W has r columns. Entries are in {-1, -0.5, 0, 0.5, 1}
    # This is a huge search space, so use a smarter approach.
    
    # Let's try: start with rank 24, and try to merge pairs of columns.
    # Two columns c1, c2 can be merged if there exist new columns that replace them
    # with the same net effect. This is hard to search directly.
    
    # Instead, let's use a gradient-free optimization approach:
    # Initialize random half-integer matrices of rank r, and iteratively adjust
    # to reduce the error.
    
    def try_rank(r, max_iters=5000):
        """Try to find a rank-r decomposition."""
        if r >= best_rank:
            return None
        np.random.seed(42 + r)
        
        for attempt in range(50):
            if time.time() > deadline:
                return None
            
            # Initialize with random small half-integer values
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            # Use a structured initialization based on schoolbook
            # Pick r of the 24 schoolbook terms
            indices = np.random.choice(n*m*p, size=r, replace=False)
            for c, idx in enumerate(indices):
                i = idx // (m*p)
                j = (idx // p) % m
                k = idx % p
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
            
            # Now try to optimize: adjust entries to make the decomposition exact
            # Use coordinate descent
            for iteration in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Compute error
                error = 0.0
                total = np.zeros((n, m, p))
                for c in range(r):
                    uc = U[:, c]
                    vc = V[:, c]
                    wc = W[:, c]
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                total[i, j, k] += uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
                
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            err = abs(total[i,j,k] - 1.0)
                            max_err = max(max_err, err)
                
                if max_err < 1e-9:
                    # Check half-integer constraint
                    if _check_half_integer_matrices(U.tolist(), V.tolist(), W.tolist()):
                        return U.tolist(), V.tolist(), W.tolist()
                    # If not half-integer, try to snap
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    if _verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p):
                        return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
                    return None
                
                # Coordinate descent: try adjusting one entry at a time
                # This is slow, so use a smarter approach
                # For each (matrix, row, col), try changing the value by +/-0.5
                matrices = [U, V, W]
                shapes = [(n*m, r), (m*p, r), (p*n, r)]
                
                # Pick a random coordinate to update
                mi = np.random.randint(3)
                mat = matrices[mi]
                ri, ci = shapes[mi]
                row_idx = np.random.randint(ri)
                col_idx = np.random.randint(r)
                
                old_val = mat[row_idx, col_idx]
                # Try nearby half-integer values
                candidates = [old_val + 0.5, old_val - 0.5, 0.0, 1.0, -1.0, 0.5, -0.5]
                candidates = list(set(candidates))
                
                best_candidate = old_val
                best_err = max_err
                
                for cand in candidates:
                    mat[row_idx, col_idx] = cand
                    # Recompute error (expensive, but necessary)
                    total = np.zeros((n, m, p))
                    for c in range(r):
                        uc = U[:, c]
                        vc = V[:, c]
                        wc = W[:, c]
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    total[i, j, k] += uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                err = max(err, abs(total[i,j,k] - 1.0))
                    if err < best_err:
                        best_err = err
                        best_candidate = cand
                    if err < 1e-9:
                        break
                
                mat[row_idx, col_idx] = best_candidate
        
        return None
    
    # Try ranks from 23 down to 12
    for r in range(23, 11, -1):
        if time.time() > deadline:
            break
        result = try_rank(r)
        if result is not None:
            U_r, V_r, W_r = result
            best_result = {"U": U_r, "V": V_r, "W": W_r}
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
