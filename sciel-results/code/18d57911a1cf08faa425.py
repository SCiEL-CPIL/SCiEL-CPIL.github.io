"""Rewritten program: search for rank-minimal exact tensor decomposition of 3x3x5 matmul."""
DIMENSIONS = [3, 3, 5]
import time
import numpy as np
from itertools import product
from fractions import Fraction

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(rank):
        # Vectorized update
        u_col = U[:, c]
        v_col = V[:, c]
        w_col = W[:, c]
        # Reshape for broadcasting
        u_reshaped = u_col.reshape(n, m)
        v_reshaped = v_col.reshape(m, p)
        w_reshaped = w_col.reshape(p, n)
        # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_reshaped[i, j] * v_reshaped[j, k] * w_reshaped[k, i]
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol)

def _verify_exact_rational(U, V, W, n, m, p):
    """Verify exact rational decomposition."""
    rank = len(U[0])
    T = np.zeros((n, m, p), dtype=object)
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i, j, k] != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook (rank = n*m*p) decomposition."""
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
    return {"U": U, "V": V, "W": W}

def _strassen_like_3x3(n, m, p):
    """
    Try to build a Strassen-like decomposition for the 3x3 block, extended to 3x5.
    For 3x3x3, the best known is rank 23 (Winograd). We use a simpler approach:
    Treat the 3x5 as 3x3 + 3x2, or use the structure directly.
    
    Actually, let's try a different approach: use the known 3x3x3 rank-23 decomposition
    and extend it. But that's complex. Instead, let's try a structured search.
    """
    pass

def _try_rank_with_als(rank, n, m, p, max_iters=200, tol=1e-8, deadline=None):
    """
    Try to find a rank-rank decomposition using ALS (Alternating Least Squares).
    The tensor is T[i,j,k] = 1 for all (i,j,k).
    We want U (nm x r), V (mp x r), W (pn x r) such that
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    if deadline is not None and time.time() > deadline:
        return None
    
    # Initialize randomly
    np.random.seed(42)
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Normalize columns
    for c in range(rank):
        norm = np.linalg.norm(U[:, c])
        if norm > 1e-10:
            U[:, c] /= norm
            V[:, c] *= norm
    for c in range(rank):
        norm = np.linalg.norm(V[:, c])
        if norm > 1e-10:
            V[:, c] /= norm
            W[:, c] *= norm
    
    best_error = float('inf')
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U: for each (i,j), solve for U[i*m+j, :]
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For fixed (i,j), we have: 1 = sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) for all k
        # This is: 1_k = U[i*m+j, :] @ (V[j*p+k, :] * W[k*n+i, :]) for each k
        # So we have p equations in r unknowns (U[i*m+j, 0..r-1])
        # Let M[k, c] = V[j*p+k, c] * W[k*n+i, c]
        # Then 1 = U[i*m+j, :] @ M^T
        # If p >= r, we can solve in least squares
        
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Solve M^T @ U[i*m+j, :] = 1
                try:
                    U[i * m + j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V: for each (j,k), solve for V[j*p+k, :]
        # 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for all i
        # Let M2[i, c] = U[i*m+j, c] * W[k*n+i, c]
        # Then 1 = V[j*p+k, :] @ M2^T
        for j in range(m):
            for k in range(p):
                M2 = np.zeros((n, rank))
                for i in range(n):
                    M2[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    V[j * p + k, :] = np.linalg.lstsq(M2.T, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W: for each (k,i), solve for W[k*n+i, :]
        # 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for all j
        # Let M3[j, c] = U[i*m+j, c] * V[j*p+k, c]
        # Then 1 = W[k*n+i, :] @ M3^T
        for k in range(p):
            for i in range(n):
                M3 = np.zeros((m, rank))
                for j in range(m):
                    M3[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    W[k * n + i, :] = np.linalg.lstsq(M3.T, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Compute error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    error += (val - 1.0) ** 2
        error = np.sqrt(error / (n * m * p))
        
        if error < best_error:
            best_error = error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if error < tol:
            break
    
    if best_error < tol:
        # Convert to integer/half-integer if possible
        U_int = [[int(round(x * 2)) / 2 for x in best_U[row]] for row in range(n * m)]
        V_int = [[int(round(x * 2)) / 2 for x in best_V[row]] for row in range(m * p)]
        W_int = [[int(round(x * 2)) / 2 for x in best_W[row]] for row in range(p * n)]
        
        if _verify_decomposition(U_int, V_int, W_int, n, m, p):
            return {"U": U_int, "V": V_int, "W": W_int}
    
    return None

def _try_structured_rank(rank, n, m, p, deadline=None):
    """
    Try a structured decomposition where each term has a specific pattern.
    For 3x3x5, we can try to use the fact that the tensor is all-ones.
    """
    if deadline is not None and time.time() > deadline:
        return None
    
    # Try a decomposition where we group terms by (i,k) pairs
    # For each (i,k), we need to compute sum_j A[i,j] * B[j,k]
    # If we use rank = n*p = 15, we'd need one term per (i,k), but that's not possible
    # with a single rank-1 term
    
    # Let's try rank = n*m*p = 45 (schoolbook) as fallback
    # For now, just return None to indicate failure
    
    return None

def _try_winograd_3x3_extended(n, m, p, deadline=None):
    """
    Try to use the Winograd algorithm for 3x3x3 and extend to 3x5.
    Winograd for 3x3x3 uses rank 15 (actually 15 for 3x3x3 is not minimal, 
    but it's a known decomposition).
    
    Actually, the best known for 3x3x3 is rank 23. Let's try to use that.
    """
    if deadline is not None and time.time() > deadline:
        return None
    
    # Winograd 3x3x3 decomposition (rank 15 is not correct, let me use a known one)
    # For now, this is a placeholder. The actual Winograd decomposition is complex.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # First, try to find a lower rank decomposition using ALS
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # But for matrix multiplication, the lower bound is higher.
    # For 3x3x3, the lower bound is 19 (actually 23 is best known).
    # For 3x3x5, let's try ranks from 15 to 45
    
    # Try ALS for various ranks
    for rank in range(15, 46):
        if time.time() > deadline:
            break
        
        result = _try_rank_with_als(rank, n, m, p, max_iters=100, tol=1e-8, deadline=deadline)
        if result is not None:
            # Verify
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                return result
    
    # If ALS failed, try the schoolbook decomposition
    result = _schoolbook(n, m, p)
    
    # Verify the schoolbook decomposition
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # If verification fails (shouldn't happen), return anyway
    return result

# EVOLVE-BLOCK-END
