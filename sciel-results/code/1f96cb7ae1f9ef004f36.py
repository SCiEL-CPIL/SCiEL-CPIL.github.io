"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def _schoolbook():
    """Return the trivial rank n*m*p decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _check_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if U, V, W form a valid decomposition of the matmul tensor."""
    # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matmul, T[i,j,k] = 1 if the (i,j,k) index corresponds to a valid multiplication term.
    # Actually, the matmul tensor is: T[i, j, k] = 1 for all i,j,k (since (AB)[i,k] = sum_j A[i,j]B[j,k])
    # Wait, the tensor for matrix multiplication is such that:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has indices (i, j, k) and T[i,j,k] = 1 for all valid i,j,k.
    # The decomposition requires: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's verify by checking a few random indices
    for _ in range(50):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        val = 0.0
        for c in range(U.shape[1]):
            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        if abs(val - 1.0) > tol:
            return False
    return True

def _try_rank(r, n, m, p, max_time=30.0, seed=42):
    """Try to find a rank-r decomposition. Returns (U, V, W) or None."""
    start_time = time.time()
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    # For rank-r decomposition, we need:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # This is a system of n*m*p = 105 equations with 3*r unknowns (well, r unknowns per factor)
    # We can try to solve this using alternating least squares or similar.
    
    # Initialize with random values
    for attempt in range(20):
        if time.time() - start_time > max_time:
            break
        
        U = np_rng.randn(n*m, r) * 0.1
        V = np_rng.randn(m*p, r) * 0.1
        W = np_rng.randn(p*n, r) * 0.1
        
        # Alternating minimization
        for it in range(200):
            if time.time() - start_time > max_time:
                break
            
            # Update U: for each (i,j), we need sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
            # This is a linear system: for fixed (i,j), let A_k = sum_c V[j*p+k,c] * W[k*n+i,c]
            # Then we need U[i*m+j, :] dot A = 1 for all k. But A depends on k...
            # Actually, for fixed (i,j), the equation is:
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for k=0..p-1
            # Let M be a p x r matrix where M[k,c] = V[j*p+k,c] * W[k*n+i,c]
            # Then we need M @ U[i*m+j, :] = ones(p)
            # Solve least squares
            
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve M @ x = ones(p)
                    try:
                        x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U_new[i*m+j, :] = x
                    except:
                        U_new[i*m+j, :] = U[i*m+j, :]
            U = U_new
            
            # Update V: for each (j,k), need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
            # Let M be an n x r matrix where M[i,c] = U[i*m+j,c] * W[k*n+i,c]
            # Then M @ V[j*p+k, :] = ones(n)
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V_new[j*p+k, :] = x
                    except:
                        V_new[j*p+k, :] = V[j*p+k, :]
            V = V_new
            
            # Update W: for each (k,i), need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
            # Let M be an m x r matrix where M[j,c] = U[i*m+j,c] * V[j*p+k,c]
            # Then M @ W[k*n+i, :] = ones(m)
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W_new[k*n+i, :] = x
                    except:
                        W_new[k*n+i, :] = W[k*n+i, :]
            W = W_new
            
            # Check convergence
            if _check_decomposition(U, V, W, n, m, p, tol=1e-4):
                # Verify more thoroughly
                if _check_decomposition(U, V, W, n, m, p, tol=1e-6):
                    return U, V, W
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix."""
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook and try to reduce rank
    # Known lower bound for matmul rank: at least n*m, m*p, p*n, and the Strassen-like bounds
    # For 3x5x7, let's try ranks from a reasonable lower bound up
    
    # Trivial rank is 105. Let's try to find something smaller.
    # The minimum rank for matmul is at least max(n, m, p) * something... 
    # Actually, the rank is at least n*m*p / (something) but for exact decomposition it's at least n*m*p in general
    # unless there's structure. For matmul, the rank is exactly n*m*p for the standard basis,
    # but there might be lower-rank decompositions using non-standard bases.
    
    # Let's try a few ranks
    best_U, best_V, best_W = _schoolbook()
    best_rank = n * m * p
    
    # Try ranks from 50 up to best_rank-1
    for r in range(50, best_rank):
        if time.time() - start_time > 110:  # Leave margin for 120s limit
            break
        result = _try_rank(r, n, m, p, max_time=10.0, seed=r)
        if result is not None:
            U, V, W = result
            # Verify
            if _check_decomposition(U, V, W, n, m, p, tol=1e-6):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                break  # Found a valid one, stop
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
