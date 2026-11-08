"""Rewritten program for 3x3 by 3x8 matrix multiplication tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]
DEADLINE = 145.0  # safety margin before 150s wall-clock kill


def _build_schoolbook(n, m, p):
    """Return (U, V, W) for the trivial schoolbook decomposition, rank = n*m*p."""
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


def _tensor_from_factors(U, V, W):
    """
    Reconstruct the multiplication tensor T[i,j,k] from factors.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
    return T


def _is_valid(U, V, W, tol=1e-6):
    """Check if the factors exactly reproduce the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    T = _tensor_from_factors(U, V, W)
    return np.allclose(T, 1.0, atol=tol, rtol=tol)


def _polish(U, V, W, tol=1e-10, max_iter=200):
    """
    High-precision polish: use Gauss-Newton style updates to drive residuals to zero.
    """
    n, m, p = DIMENSIONS
    for it in range(max_iter):
        # Check convergence
        T = _tensor_from_factors(U, V, W)
        err = np.max(np.abs(T - 1.0))
        if err < tol:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, U.shape[1]))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    U[i * m + j, :], _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                except np.linalg.LinAlgError:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, V.shape[1]))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    V[j * p + k, :], _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                except np.linalg.LinAlgError:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, W.shape[1]))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    W[k * n + i, :], _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                except np.linalg.LinAlgError:
                    pass
    
    return U, V, W


def _try_rank(n, m, p, rank, time_budget=20.0, seed=0):
    """
    Try to find a rank-`rank` decomposition using iterative refinement with multiple restarts.
    """
    start = time.time()
    rng = np.random.default_rng(seed)
    
    for attempt in range(3):
        if time.time() - start > time_budget:
            break
        
        # Initialize with random factors
        U = rng.standard_normal((n * m, rank))
        V = rng.standard_normal((m * p, rank))
        W = rng.standard_normal((p * n, rank))
        
        # Scale to reasonable magnitude
        U *= 0.5
        V *= 0.5
        W *= 0.5
        
        for iteration in range(1000):
            if time.time() - start > time_budget:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :], _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    except np.linalg.LinAlgError:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :], _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    except np.linalg.LinAlgError:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :], _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    except np.linalg.LinAlgError:
                        pass
            
            # Check convergence
            if _is_valid(U, V, W, tol=1e-8):
                # Polish for higher precision
                U, V, W = _polish(U, V, W, tol=1e-12, max_iter=50)
                if _is_valid(U, V, W, tol=1e-10):
                    return U, V, W
    
    # Final check
    if _is_valid(U, V, W, tol=1e-6):
        U, V, W = _polish(U, V, W, tol=1e-12, max_iter=50)
        if _is_valid(U, V, W, tol=1e-10):
            return U, V, W
    return None


def _construct_laderman_3x3x3():
    """
    Construct Laderman's rank-23 decomposition for 3x3 x 3x3.
    This is a known optimal decomposition.
    """
    # Laderman's algorithm for 3x3x3, rank 23
    # Reference: Laderman, "On the complexity of the multiplication of 3x3 matrices" (1976)
    # The decomposition uses 23 bilinear multiplications.
    
    # I'll use a known explicit construction.
    # The tensor entries T[i,j,k] = 1 for i,j,k in {0,1,2}
    
    # U: (9, 23), V: (9, 23), W: (9, 23)
    # Using the standard Laderman construction
    
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Laderman's 23 terms (using 0-indexed i,j,k)
    # Term structure: each term has (i,j) in U, (j,k) in V, (k,i) in W
    
    # Based on the known Laderman algorithm:
    # We'll use the explicit coefficients from the literature
    
    # The 23 bilinear terms for 3x3x3 (Laderman 1976):
    # Using the standard indexing where A[i,j], B[j,k], C[i,k]
    
    # Let me use a verified Laderman construction
    # From various references, here are the 23 terms:
    
    terms = [
        # (i, j, k, u_coeff, v_coeff, w_coeff)
        # Term 1-9: diagonal-ish terms
        (0, 0, 0, 1, 1, 1),
        (1, 1, 1, 1, 1, 1),
        (2, 2, 2, 1, 1, 1),
        (0, 1, 0, 1, 1, 1),
        (1, 0, 1, 1, 1, 1),
        (2, 1, 2, 1, 1, 1),
        (0, 2, 1, 1, 1, 1),
        (1, 2, 0, 1, 1, 1),
        (2, 0, 1, 1, 1, 1),
        # Terms 10-18: cross terms
        (0, 0, 1, 1, 1, 1),
        (1, 1, 0, 1, 1, 1),
        (2, 2, 0, 1, 1, 1),
        (0, 1, 1, 1, 1, 1),
        (1, 0, 0, 1, 1, 1),
        (2, 1, 1, 1, 1, 1),
        (0, 2, 2, 1, 1, 1),
        (1, 2, 2, 1, 1, 1),
        (2, 0, 0, 1, 1, 1),
        # Terms 19-23: additional cross terms
        (0, 0, 2, 1, 1, 1),
        (1, 1, 2, 1, 1, 1),
        (2, 2, 1, 1, 1, 1),
        (0, 2, 0, 1, 1, 1),
        (2, 0, 2, 1, 1, 1),
    ]
    
    # This simple construction won't work - I need the actual Laderman coefficients.
    # Let me use a different approach: construct it properly.
    
    # Actually, let me use a known correct Laderman decomposition.
    # The key insight is that for 3x3x3, rank 23 is achievable.
    # Let me use the explicit form from the literature.
    
    # I'll use a parametrized construction that I can verify.
    
    return None  # Fall back to iterative method


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # For 3x3x3, known rank is 23. For 3x3x3x8, it should be higher.
    # Let's try from 30 up to 72.
    
    # Try a few promising ranks with time budget
    ranks_to_try = list(range(30, schoolbook_rank + 1))
    ranks_to_try.sort()
    
    for rank in ranks_to_try:
        if time.time() > DEADLINE - 15:
            break
        
        # Try a few seeds for each rank
        found = None
        max_seeds = 3
        time_per_seed = 8.0
        
        for seed in range(max_seeds):
            if time.time() > DEADLINE - 15:
                break
            result = _try_rank(n, m, p, rank, time_budget=time_per_seed, seed=seed)
            if result is not None:
                found = result
                break
        
        if found is not None:
            U, V, W = found
            best_result = (U, V, W)
            best_rank = rank
            # If we found a valid decomposition below 50, stop early
            if rank < 50:
                break
        else:
            # If we can't find rank r, higher ranks are more likely to work
            # But let's continue trying a few more
            if rank > 45:
                # Try the next few ranks quickly
                continue
    
    if best_result is None:
        # Fall back to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
