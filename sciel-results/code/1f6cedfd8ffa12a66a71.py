"""Tensor decomposition for 2x3x6 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def _verify_exact(U, V, W, n=2, m=3, p=6, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook():
    """Return the trivial rank-36 schoolbook decomposition."""
    n, m, p = 2, 3, 6
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _alternating_minimize(U, V, W, n=2, m=3, p=6, max_iter=200, tol=1e-10):
    """
    Alternating least squares for tensor decomposition.
    T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    Target: T[i,j,k] = 1 for all valid (i,j,k).
    
    We solve for each factor in turn, holding the other two fixed.
    For U: fix V, W. For each (i,j), we need:
        sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
    This is a linear system in U[i*m+j,:].
    Similarly for V and W.
    """
    r = len(U[0]) if len(U) > 0 else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for it in range(max_iter):
        # Update U
        for i in range(n):
            for j in range(m):
                # Build system: sum_c U[i*m+j,c] * M[c] = 1 for each k
                # M[c] = V[j*p+k,c] * W[k*n+i,c] for each k
                # We have p equations in r unknowns (U[i*m+j, 0..r-1])
                # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                A_mat = np.zeros((p, r))
                for k in range(p):
                    A_mat[k] = V[j*p+k] * W[k*n+i]
                # Solve A_mat @ U[i*m+j] = ones(p)
                # Least squares
                sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                U[i*m+j] = sol
        
        # Update V
        for j in range(m):
            for k in range(p):
                # sum_c V[j*p+k,c] * N[c] = 1 for each i
                # N[c] = U[i*m+j,c] * W[k*n+i,c]
                A_mat = np.zeros((n, r))
                for i in range(n):
                    A_mat[i] = U[i*m+j] * W[k*n+i]
                sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
                V[j*p+k] = sol
        
        # Update W
        for k in range(p):
            for i in range(n):
                # sum_c W[k*n+i,c] * P[c] = 1 for each j
                # P[c] = U[i*m+j,c] * V[j*p+k,c]
                A_mat = np.zeros((m, r))
                for j in range(m):
                    A_mat[j] = U[i*m+j] * V[j*p+k]
                sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                W[k*n+i] = sol
        
        # Check convergence
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    max_err = max(max_err, abs(val - 1.0))
        if max_err < tol:
            break
    
    return U, V, W, max_err

def _try_rank(r, n=2, m=3, p=6, n_attempts=30, deadline=None):
    """Try to find a rank-r decomposition via alternating minimization."""
    best_err = float('inf')
    best = None
    
    for attempt in range(n_attempts):
        if deadline and time.time() > deadline - 2:
            break
        
        # Random initialization with some structure
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # Normalize to help convergence
        for c in range(r):
            norm_u = np.linalg.norm(U[:, c])
            norm_v = np.linalg.norm(V[:, c])
            norm_w = np.linalg.norm(W[:, c])
            if norm_u > 1e-12 and norm_v > 1e-12 and norm_w > 1e-12:
                scale = (norm_u * norm_v * norm_w) ** (1.0/3.0)
                U[:, c] /= scale
                V[:, c] /= scale
                W[:, c] /= scale
        
        U, V, W, err = _alternating_minimize(U, V, W, n, m, p, max_iter=100)
        
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
            if err < 1e-8:
                break
    
    return best, best_err

def _structured_2x3x6():
    """
    Try to exploit structure: view 2x3x6 as related to 2x3x3.
    If we can find a rank-r decomposition for 2x3x3, we might extend it.
    
    For 2x3x3, the tensor T[i,j,k]=1 for i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    Known: rank of 2x3x3 multiplication is at most 6 (schoolbook), possibly lower.
    
    For 2x3x6, write k = 3*q + k' where q in {0,1}, k' in {0,1,2}.
    T[i,j,3q+k'] = 1.
    
    If we have a rank-r decomposition for the 2x3x3 case with factors U0, V0, W0,
    then for the 6-column case, we need two copies (one for each q), but we might
    share the U factor. This gives rank 2r for the V and W factors sharing U.
    But that's not better than schoolbook (36 = 2*3*6).
    
    Better idea: try to find direct low-rank decompositions using search.
    """
    n, m, p = 2, 3, 6
    # Start with a structured initialization based on Kronecker products
    # Try rank 18 (half of schoolbook) with structured init
    r = 18
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    # Structured: split into two 2x3x3 blocks
    for q in range(2):
        for i in range(n):
            for j in range(m):
                for k in range(3):
                    c = q * 9 + i * 3 + j
                    U[i*m+j, c] = 1.0
                    V[j*p + q*3 + k, c] = 1.0
                    W[(q*3+k)*n + i, c] = 1.0
    
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # Safety margin
    
    n, m, p = 2, 3, 6
    schoolbook_result = _schoolbook()
    
    best_result = schoolbook_result
    best_rank = n * m * p  # 36
    
    # Try ranks from 18 up to 35
    for r in range(18, min(36, best_rank)):
        if time.time() > deadline - 10:
            break
        
        # Try structured initialization first
        if r == 18:
            U0, V0, W0 = _structured_2x3x6()
            U, V, W, err = _alternating_minimize(U0, V0, W0, n, m, p, max_iter=300)
            if err < 1e-8:
                best_rank = r
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                if time.time() > deadline - 5:
                    break
                continue
        
        # Try random initializations
        result, err = _try_rank(r, n, m, p, n_attempts=15, deadline=deadline)
        if result is not None and err < 1e-8:
            U, V, W = result
            best_rank = r
            best_result = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            if time.time() > deadline - 5:
                break
    
    # Final verification
    if best_result is not schoolbook_result:
        if not _verify_exact(best_result["U"], best_result["V"], best_result["W"], n, m, p):
            # Fall back to schoolbook if verification fails
            best_result = schoolbook_result
    
    return best_result
# EVOLVE-BLOCK-END
