"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_START_TIME = time.time()
_DEADLINE = 150.0  # wall-clock limit

def _elapsed():
    return time.time() - _START_TIME

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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
    return U, V, W

def _to_half_integer(mat):
    """Round matrix entries to nearest half-integer (0.5 steps)."""
    return np.round(2.0 * mat) / 2.0

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly reproduces matrix multiplication."""
    # The multiplication tensor T[i,j,k] = 1 if we compute (AB)[i,k] = sum_j A[i,j] B[j,k]
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all (i,j,k)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_t = U.reshape(n, m, -1)
    V_t = V.reshape(m, p, -1)
    W_t = W.reshape(p, n, -1)
    # Contract over rank: result[i,j,k] = sum_r U_t[i,j,r] * V_t[j,p,r] * W_t[k,n,r]
    # Use einsum: 'ijr,jkr,kin->ijk'
    result = np.einsum('ijr,jkr,kin->ijk', U_t, V_t, W_t)
    # Should be all ones
    return np.allclose(result, 1.0, atol=1e-6)

def _try_rank(r, n, m, p, max_iters=2000, seed=42):
    """Try to find a rank-r decomposition via randomized initialization + ALS-style updates."""
    if r >= n * m * p:
        U, V, W = _schoolbook(n, m, p)
        return U, V, W, True
    
    rng = np.random.RandomState(seed)
    best_error = float('inf')
    best_U = best_V = best_W = None
    
    for attempt in range(3):
        if _elapsed() > _DEADLINE - 10:
            break
        
        # Initialize
        U = rng.randn(n * m, r) * 0.5
        V = rng.randn(m * p, r) * 0.5
        W = rng.randn(p * n, r) * 0.5
        
        # Normalize columns
        for mat in (U, V, W):
            norms = np.linalg.norm(mat, axis=0)
            norms[norms < 1e-12] = 1e-12
            mat /= norms
        
        for it in range(max_iters):
            if _elapsed() > _DEADLINE - 10:
                break
            
            # Update U: for each (i,j), U[i*m+j, :] should satisfy
            # sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for all k
            # This is a linear system in U[i*m+j, :]
            # Let's build the matrix: for fixed (i,j), we need
            # U_row @ (V[j,:] outer W[:,i])^T ... hmm, let me think differently.
            
            # Alternative: treat as tensor completion / ALS on the 3-tensor.
            # For fixed (i,j): minimize ||sum_r U[i,j,r] * V[j,:,r] * W[:,i,r] - 1||^2
            # This is linear in U[i,j,:]. Let M[i,j] be the matrix where
            # M[i,j][k, r] = V[j*p+k, r] * W[k*n+i, r]
            # Then we want M[i,j] @ U[i*m+j, :] = ones(p)
            
            # Similarly for V[j,:,r] with fixed (i,j) and for W[:,i,r] with fixed (i,k).
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # Build M: shape (p, r)
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve least squares: M @ x = ones(p)
                    try:
                        x, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    except np.linalg.LinAlgError:
                        x = np.zeros(r)
                    U[i * m + j, :] = x
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        x, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    except np.linalg.LinAlgError:
                        x = np.zeros(r)
                    V[j * p + k, :] = x
            
            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        x, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    except np.linalg.LinAlgError:
                        x = np.zeros(r)
                    W[k * n + i, :] = x
            
            # Check error
            U_t = U.reshape(n, m, -1)
            V_t = V.reshape(m, p, -1)
            W_t = W.reshape(p, n, -1)
            result = np.einsum('ijr,jkr,kin->ijk', U_t, V_t, W_t)
            err = np.max(np.abs(result - 1.0))
            
            if err < best_error:
                best_error = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-8:
                break
        
        if best_error < 1e-8:
            # Try to snap to half-integers
            U_hi = _to_half_integer(best_U)
            V_hi = _to_half_integer(best_V)
            W_hi = _to_half_integer(best_W)
            if _verify(U_hi, V_hi, W_hi, n, m, p):
                return U_hi, V_hi, W_hi, True
            # Also try rounding to integers
            U_int = np.round(best_U)
            V_int = np.round(best_V)
            W_int = np.round(best_W)
            if _verify(U_int, V_int, W_int, n, m, p):
                return U_int, V_int, W_int, True
    
    if best_U is not None and best_error < 1e-8:
        return best_U, best_V, best_W, True
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    
    # Try to find lower-rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(16, 28, 28) = 28
    # Upper bound: 112 (schoolbook)
    # Try ranks from a reasonable lower bound upward
    
    # For 4x4x7, the schoolbook is 112.
    # Let's try some candidate ranks.
    # We'll try a few ranks and see if we can find a valid decomposition.
    
    candidate_ranks = [28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 111]
    
    # Sort by rank (ascending) to try smallest first
    candidate_ranks = sorted(set(candidate_ranks))
    
    for r in candidate_ranks:
        if _elapsed() > _DEADLINE - 15:
            break
        if r >= n * m * p:
            continue
        U, V, W, success = _try_rank(r, n, m, p, max_iters=500, seed=r * 7 + 1)
        if success:
            # Convert to list of lists
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
# EVOLVE-BLOCK-END
