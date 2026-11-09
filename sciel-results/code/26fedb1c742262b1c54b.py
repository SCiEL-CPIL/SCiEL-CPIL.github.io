"""Direct approach for 3x3 by 3x6 matrix multiplication tensor decomposition.

Strategy:
  1. Attempt a rank-9 structured decomposition exploiting row-wise structure.
  2. Verify against the target tensor.
  3. Fall back to schoolbook (rank 54) if verification fails.
"""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]
TIME_LIMIT = 115  # safety margin before 120s kill


def _build_rank9(n=3, m=3, p=6):
    """Construct rank-9 decomposition attempt."""
    rank = n * m
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            U[i * m + j, r] = 1.0
            for k in range(p):
                V[j * p + k, r] = 1.0
            for k2 in range(p):
                W[k2 * n + i, r] = 1.0
    
    return U, V, W


def _build_schoolbook(n=3, m=3, p=6):
    """Return (U, V, W) for the schoolbook decomposition, rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
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
    """Reconstruct the tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]."""
    a_dim = U.shape[0]
    b_dim = V.shape[0]
    c_dim = W.shape[0]
    r = U.shape[1]
    T = np.zeros((a_dim, b_dim, c_dim), dtype=np.float64)
    for r_idx in range(r):
        T += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
    return T


def _target_tensor(n=3, m=3, p=6):
    """Build the target multiplication tensor."""
    a_dim = n * m
    b_dim = m * p
    c_dim = p * n
    T = np.zeros((a_dim, b_dim, c_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _verify(U, V, W, target, tol=1e-9):
    """Check that the factorisation reproduces the target tensor exactly."""
    T = _tensor_from_factors(U, V, W)
    return np.allclose(T, target, atol=tol, rtol=tol)


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try rank-9 decomposition first
    U9, V9, W9 = _build_rank9(n, m, p)
    target = _target_tensor(n, m, p)
    
    if _verify(U9, V9, W9, target):
        return {"U": U9.tolist(), "V": V9.tolist(), "W": W9.tolist()}
    
    # Fallback to schoolbook
    U_fb, V_fb, W_fb = _build_schoolbook(n, m, p)
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}


# EVOLVE-BLOCK-START
# (construct is already defined above; this block is kept for interface compatibility)
# EVOLVE-BLOCK-END
