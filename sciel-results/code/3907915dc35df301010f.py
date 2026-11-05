"""Matrix multiplication 4,4,5 - rank-minimal tensor decomposition search."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [4, 4, 5]
n, m, p = 4, 4, 5

def construct():
    start_time = time.time()
    deadline = 115.0  # seconds, safety margin before 120s limit
    
    # Target tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need U (n*m x r), V (m*p x r), W (p*n x r) such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    best_rank = n * m * p  # schoolbook fallback
    best_factors = _schoolbook(n, m, p)
    
    # Try structured low-rank decompositions
    # Strategy: use bilinear forms with structured support patterns
    
    # Attempt 1: Block-structured decomposition
    # Partition indices into blocks and use Kronecker-like products
    for r in range(36, 80):
        if time.time() - start_time > deadline:
            break
        result = _try_structured(n, m, p, r)
        if result is not None:
            best_rank = r
            best_factors = result
            break
    
    # Attempt 2: Random initialization + local refinement for smaller ranks
    if time.time() - start_time < deadline - 10:
        for r in range(30, min(45, best_rank)):
            if time.time() - start_time > deadline:
                break
            result = _random_search(n, m, p, r, max_iters=5000)
            if result is not None:
                best_rank = r
                best_factors = result
                break
    
    # Attempt 3: If we found something, try to polish
    if best_rank < n * m * p and time.time() - start_time < deadline - 5:
        polished = _polish(n, m, p, best_factors, best_rank)
        if polished is not None:
            best_factors = polished
    
    return {"U": best_factors[0], "V": best_factors[1], "W": best_factors[2]}


def _schoolbook(n, m, p):
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
    return (U, V, W)


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if decomposition is exact (within tolerance)."""
    r = U.shape[1]
    # Compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Reshape for efficient computation
    U_r = U.reshape(n, m, r)
    V_r = V.reshape(m, p, r)
    W_r = W.reshape(p, n, r)
    
    # Result[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # This is a tensor contraction
    result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    
    # Should be all ones
    max_err = np.max(np.abs(result - 1.0))
    return max_err < tol, max_err


def _try_structured(n, m, p, r):
    """Try structured decomposition with specific support patterns."""
    # Use a pattern-based approach: each rank-1 component has specific support
    # Try using the "laser method" style or structured bilinear forms
    
    # For 4x4x5, try using known constructions
    # One approach: embed in 5x5x5 and project
    
    # Try: use r components where each has structured support
    # Pattern: for component c, define support sets S_c^U, S_c^V, S_c^W
    
    # Simple structured attempt: use Kronecker product structure
    # 4 = 2*2, so try 2x2 block structure
    
    # Actually, let's try a more direct approach:
    # Initialize with random sparse patterns and optimize
    
    np.random.seed(42)
    
    # Try multiple random initializations
    for trial in range(20):
        # Initialize with random values
        U = np.random.randn(n * m, r) * 0.5
        V = np.random.randn(m * p, r) * 0.5
        W = np.random.randn(p * n, r) * 0.5
        
        # Try to optimize
        success, final_U, final_V, final_W = _local_optimize(U, V, W, n, m, p, max_iters=2000)
        if success:
            return (final_U, final_V, final_W)
    
    return None


def _local_optimize(U, V, W, n, m, p, max_iters=5000, tol=1e-8):
    """Alternating least squares / gradient-based optimization."""
    r = U.shape[1]
    
    # Precompute reshapes
    U_r = U.reshape(n, m, r)
    V_r = V.reshape(m, p, r)
    W_r = W.reshape(p, n, r)
    
    for it in range(max_iters):
        # Update U: for each (i,j), minimize sum_k (1 - sum_c U[i,j,c]*V[j,k,c]*W[k,i,c])^2
        # This is a linear system in U[i,j,:]
        for i in range(n):
            for j in range(m):
                # A = V[j,:,:] * W[:,i,:] element-wise, shape (p, r)
                A = V_r[j, :, :] * W_r[:, i, :]  # (p, r)
                # b = ones(p)
                b = np.ones(p)
                # Solve A^T A * x = A^T b
                try:
                    x, residuals, rank_s, sv = np.linalg.lstsq(A, b, rcond=None)
                except:
                    continue
                U_r[i, j, :] = x
        
        # Update V: for each (j,k), minimize sum_i (1 - sum_c U[i,j,c]*V[j,k,c]*W[k,i,c])^2
        for j in range(m):
            for k in range(p):
                A = U_r[:, j, :] * W_r[k, :, :]  # (n, r)
                b = np.ones(n)
                try:
                    x, residuals, rank_s, sv = np.linalg.lstsq(A, b, rcond=None)
                except:
                    continue
                V_r[j, k, :] = x
        
        # Update W: for each (k,i), minimize sum_j (1 - sum_c U[i,j,c]*V[j,k,c]*W[k,i,c])^2
        for k in range(p):
            for i in range(n):
                A = U_r[i, :, :] * V_r[:, k, :]  # (m, r)
                b = np.ones(m)
                try:
                    x, residuals, rank_s, sv = np.linalg.lstsq(A, b, rcond=None)
                except:
                    continue
                W_r[k, i, :] = x
        
        # Check convergence
        if it % 100 == 0 or it == max_iters - 1:
            U = U_r.copy()
            V = V_r.copy()
            W = W_r.copy()
            ok, err = _verify(U, V, W, n, m, p)
            if ok:
                return True, U, V, W
    
    # Final check
    U = U_r.copy()
    V = V_r.copy()
    W = W_r.copy()
    ok, err = _verify(U, V, W, n, m, p)
    return ok, U, V, W


def _random_search(n, m, p, r, max_iters=5000, n_trials=50):
    """Random initialization with local optimization."""
    for trial in range(n_trials):
        np.random.seed(trial * 12345 + 67890)
        U = np.random.randn(n * m, r) * 0.3
        V = np.random.randn(m * p, r) * 0.3
        W = np.random.randn(p * n, r) * 0.3
        
        success, final_U, final_V, final_W = _local_optimize(U, V, W, n, m, p, max_iters=max_iters)
        if success:
            return (final_U, final_V, final_W)
    
    return None


def _polish(n, m, p, factors, r):
    """High-precision polish pass."""
    U, V, W = factors
    success, final_U, final_V, final_W = _local_optimize(U, V, W, n, m, p, max_iters=10000, tol=1e-12)
    if success:
        return (final_U, final_V, final_W)
    return None
