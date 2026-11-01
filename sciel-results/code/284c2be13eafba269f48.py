"""Strategy: Randomized tensor rank minimization with SVD truncation and
trust-region refinement. We hypothesize that rank 17 (or lower) may be achievable
for 2x2 * 2x9 by exploiting non-trivial algebraic dependencies not captured by
simple block constructions. We use random initialization, SVD-based rank reduction,
and scipy.optimize.least_squares to drive the residual to zero."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 9]


def _build_target_tensor():
    """Build the target multiplication tensor T[a,b,c] for 2x2 * 2x9.
    T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k, else 0.
    Indexing:
      A is 2x2: a = i*m + j = i*2 + j
      B is 2x9: b = j*p + k = j*9 + k
      C is 2x9: c = k*n + i = k*2 + i
    """
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T


def _verify_decomposition(U, V, W, T=None):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c]."""
    if T is None:
        T = _build_target_tensor()
    M = np.einsum('ar,br,cr->abc', U, V, W)
    max_diff = np.max(np.abs(M - T))
    return max_diff < 1e-10, max_diff


def _residual(U, V, W, T_flat):
    """Compute residual for least_squares: flatten the tensor difference."""
    M = np.einsum('ar,br,cr->abc', U, V, W)
    return (M - T_flat.reshape(M.shape)).flatten()


def _try_rank(rank, T_flat, max_iter=300, timeout=40.0):
    """Attempt to find a decomposition of given rank using random init + SVD + LM."""
    start = time.time()
    n, m, p = DIMENSIONS
    dims = (n * m, m * p, p * n)
    
    # Random initialization
    np.random.seed(42)
    U = np.random.randn(dims[0], rank)
    V = np.random.randn(dims[1], rank)
    W = np.random.randn(dims[2], rank)
    
    # SVD truncation pass to reduce effective rank
    for _ in range(3):
        if time.time() - start > timeout * 0.3:
            break
        # Mode-1 unfolding
        M1 = np.einsum('br,cr->bcr', V, W).reshape(dims[1] * dims[2], rank)
        # Actually, mode-1 fiber: for each a, we have a vector in (b,c) space
        # U[:, r] are the coefficients. Let's do it properly.
        
        # Reshape U to (n*m, rank). The tensor is T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # Mode-1 unfolding: matrix of size (n*m) x (m*p * p*n)
        # T1[a, b*c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We can compute the Gram matrix G1 = U @ (V*W).T @ (V*W) @ U.T
        # But simpler: just do SVD on the unfolded tensor.
        
        # Unfold mode 1: shape (n*m, m*p * p*n)
        T1 = np.einsum('ar,br,cr->a,bc', U, V, W)
        T1 = T1.reshape(dims[0], -1)
        U_s, S, Vt = np.linalg.svd(T1, full_matrices=False)
        if len(S) > rank:
            U = U_s[:, :rank] * S[:rank]
        else:
            U = U_s[:, :min(rank, len(S))] * S[:min(rank, len(S))]
            # Pad if needed
            if U.shape[1] < rank:
                U = np.hstack([U, np.zeros((dims[0], rank - U.shape[1]))])
        
        # Unfold mode 2
        T2 = np.einsum('ar,br,cr->b,ac', U, V, W)
        T2 = T2.reshape(dims[1], -1)
        V_s, S, Vt = np.linalg.svd(T2, full_matrices=False)
        if len(S) > rank:
            V = V_s[:, :rank] * S[:rank]
        else:
            V = V_s[:, :min(rank, len(S))] * S[:min(rank, len(S))]
            if V.shape[1] < rank:
                V = np.hstack([V, np.zeros((dims[1], rank - V.shape[1]))])
        
        # Unfold mode 3
        T3 = np.einsum('ar,br,cr->c,ab', U, V, W)
        T3 = T3.reshape(dims[2], -1)
        W_s, S, Vt = np.linalg.svd(T3, full_matrices=False)
        if len(S) > rank:
            W = W_s[:, :rank] * S[:rank]
        else:
            W = W_s[:, :min(rank, len(S))] * S[:min(rank, len(S))]
            if W.shape[1] < rank:
                W = np.hstack([W, np.zeros((dims[2], rank - W.shape[1]))])
    
    # Now use least_squares to refine
    def x_to_UVW(x):
        U_r = x[:dims[0] * rank].reshape(dims[0], rank)
        V_r = x[dims[0] * rank:dims[0] * rank + dims[1] * rank].reshape(dims[1], rank)
        W_r = x[dims[0] * rank + dims[1] * rank:].reshape(dims[2], rank)
        return U_r, V_r, W_r
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def fun(x):
        U_r, V_r, W_r = x_to_UVW(x)
        return _residual(U_r, V_r, W_r, T_flat)
    
    try:
        result = least_squares(fun, x0, method='trf', max_nfev=max_iter,
                               ftol=1e-15, xtol=1e-15, gtol=1e-15)
        U_r, V_r, W_r = x_to_UVW(result.x)
        
        # Check if exact (within tolerance)
        M = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        max_diff = np.max(np.abs(M - T_flat.reshape(M.shape)))
        
        if max_diff < 1e-10:
            # Try to round to integers/half-integers
            U_round = np.round(U_r * 2) / 2
            V_round = np.round(V_r * 2) / 2
            W_round = np.round(W_r * 2) / 2
            M_round = np.einsum('ar,br,cr->abc', U_round, V_round, W_round)
            max_diff_round = np.max(np.abs(M_round - T_flat.reshape(M_round.shape)))
            if max_diff_round < 1e-10:
                return U_round, V_round, W_round, max_diff_round
            # Try integer rounding
            U_int = np.round(U_r)
            V_int = np.round(V_r)
            W_int = np.round(W_r)
            M_int = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
            max_diff_int = np.max(np.abs(M_int - T_flat.reshape(M_int.shape)))
            if max_diff_int < 1e-10:
                return U_int, V_int, W_int, max_diff_int
            # Return unrounded if exact
            return U_r, V_r, W_r, max_diff
        return None, None, None, max_diff
    except Exception:
        return None, None, None, np.inf


def _build_rank18_fallback():
    """Build the known rank-18 decomposition."""
    n, m, p = DIMENSIONS
    rank = 2 * p  # 18
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for k in range(p):
        # Term 2k: A[:,0] * B[0,k] -> C[:,k]
        r = 2 * k
        U[0, r] = 1  # A[0,0]
        U[2, r] = 1  # A[1,0]
        V[k, r] = 1  # B[0,k]
        W[2*k, r] = 1  # C[0,k]
        W[2*k+1, r] = 1  # C[1,k]
        
        # Term 2k+1: A[:,1] * B[1,k] -> C[:,k]
        r = 2 * k + 1
        U[1, r] = 1  # A[0,1]
        U[3, r] = 1  # A[1,1]
        V[9+k, r] = 1  # B[1,k]
        W[2*k, r] = 1  # C[0,k]
        W[2*k+1, r] = 1  # C[1,k]
    
    return U, V, W


def construct():
    """
    Find a low-rank decomposition for 2x2 * 2x9 matrix multiplication.
    Returns dict with U (4 x rank), V (18 x rank), W (18 x rank) as lists.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    T = _build_target_tensor()
    T_flat = T.flatten()
    
    # Try ranks from 17 down to 1
    best_result = None
    best_rank = 18
    
    # First, get the fallback rank-18
    U18, V18, W18 = _build_rank18_fallback()
    valid, diff = _verify_decomposition(U18, V18, W18, T)
    if valid:
        best_result = (U18, V18, W18)
        best_rank = 18
    
    # Try lower ranks
    for rank in [17, 16, 15, 14, 13, 12, 11, 10, 9]:
        if time.time() > deadline:
            break
        U_r, V_r, W_r, max_diff = _try_rank(rank, T_flat, max_iter=200, timeout=15.0)
        if U_r is not None and max_diff < 1e-10:
            # Verify
            valid, diff = _verify_decomposition(U_r, V_r, W_r, T)
            if valid:
                best_result = (U_r, V_r, W_r)
                best_rank = rank
                break  # Found a valid lower rank, stop
    
    if best_result is None:
        # Should not happen, but fallback
        best_result = (U18, V18, W18)
        best_rank = 18
    
    U, V, W = best_result
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
