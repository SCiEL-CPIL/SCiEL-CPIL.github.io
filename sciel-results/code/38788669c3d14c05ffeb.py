"""Tensor decomposition search for 3x4 * 4x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, tol=1e-6):
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=tol)

def _compute_error(U, V, W):
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return np.max(np.abs(T - 1.0))

def _als_step(U, V, W, n, m, p, rank):
    """One round of alternating least squares updates."""
    # Update V
    for j in range(m):
        for k in range(p):
            # M_k[i,c] = U[i*m+j,c] * W[k*n+i,c]
            M_k = np.outer(U[:, j].reshape(n, -1)[:, :rank], W[k*n:(k+1)*n, :].T)
            try:
                V[j*p+k, :] = np.linalg.lstsq(M_k.T, np.ones(n), rcond=None)[0]
            except:
                pass
    
    # Update W
    for k in range(p):
        for i in range(n):
            M_i = np.outer(U[i*m:(i+1)*m, :].T, V[:, k].reshape(m, -1)[:, :rank])
            try:
                W[k*n+i, :] = np.linalg.lstsq(M_i.T, np.ones(m), rcond=None)[0]
            except:
                pass
    
    # Update U
    for i in range(n):
        for j in range(m):
            M = V[j*p:(j+1)*p, :] * W[k*n+i, :]  # shape (p, rank)
            # Fix: need to use correct k index - this is wrong, let's fix
            # Actually for U[i*m+j,:], we need: for each k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # M[k,c] = V[j*p+k,c] * W[k*n+i,c]
            M = np.zeros((p, rank))
            for k in range(p):
                M[k, :] = V[j*p+k, :] * W[k*n+i, :]
            try:
                U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
            except:
                pass
    return U, V, W

def _try_rank(rank, seed=42, max_iters=200, deadline=None):
    """Try to find a rank-rank decomposition using structured random search with ALS."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Initialize with structured random values
        U = rng.randn(n * m, rank) * 0.3
        V = rng.randn(m * p, rank) * 0.3
        W = rng.randn(p * n, rank) * 0.3
        
        # Run ALS iterations
        for r in range(50):
            if deadline and time.time() > deadline:
                break
            U, V, W = _als_step(U, V, W, n, m, p, rank)
        
        err = _compute_error(U, V, W)
        
        if err < best_err:
            best_err = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if best_err < 1e-8:
            break
    
    if best_err < 1e-6:
        # Snap to half-integers
        best_U = np.round(best_U * 2) / 2
        best_V = np.round(best_V * 2) / 2
        best_W = np.round(best_W * 2) / 2
        err_after = _compute_error(best_U, best_V, best_W)
        if err_after < 1e-6:
            return best_U, best_V, best_W
    
    return None

def _structured_init(rank, seed=42):
    """Create structured initial guess based on block decomposition."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Use a simple structured pattern as starting point
    for c in range(rank):
        # Distribute non-zero entries
        for i in range(n):
            for j in range(m):
                if (i + j + c) % 3 == 0:
                    U[i*m+j, c] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
        for j in range(m):
            for k in range(p):
                if (j + k + c) % 3 == 0:
                    V[j*p+k, c] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
        for k in range(p):
            for i in range(n):
                if (k + i + c) % 3 == 0:
                    W[k*n+i, c] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
    
    return U, V, W

def construct():
    deadline = time.time() + 110  # Safety margin
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook()
    
    # Try ranks from lower bound upward
    # Known lower bounds for 3x4x6 suggest rank can be significantly below 72
    for rank in range(55, 72):
        if time.time() > deadline - 5:
            break
        
        # Try multiple seeds for this rank
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            result = _try_rank(rank, seed=seed + rank * 100, max_iters=50, deadline=deadline - 5)
            if result is not None:
                U, V, W = result
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fall back to schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
