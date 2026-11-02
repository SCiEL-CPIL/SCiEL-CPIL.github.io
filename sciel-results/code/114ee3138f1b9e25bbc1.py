"""Matrix multiplication 2x3 by 3x3: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]
_START = time.time()
_DEADLINE = 145.0  # seconds, safety margin

def _elapsed():
    return time.time() - _START

def _remaining():
    return _DEADLINE - _elapsed()

def _verify(U, V, W):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all (i,j,k)."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T = np.ones((n, m, p))
    recon = np.zeros((n, m, p))
    for c in range(rank):
        u = U[:, c]
        v = V[:, c]
        w = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    if not np.allclose(recon, T, atol=1e-8):
        return False
    return True

def _als_decompose(rank, n, m, p, max_iter=500, tol=1e-12, seed=0):
    """ALS optimization to find rank-r decomposition of the all-ones tensor."""
    rng = np.random.RandomState(seed)
    # Initialize with structured pattern
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Use a structured initialization based on the index pattern
    for c in range(rank):
        # Assign each factor to cover certain indices
        i_idx = c % n
        j_idx = (c // n) % m
        k_idx = (c // (n * m)) % p
        
        U[i_idx * m + j_idx, c] = 1.0
        V[j_idx * p + k_idx, c] = 1.0
        W[k_idx * n + i_idx, c] = 1.0
    
    # Add small random perturbation to break symmetry
    U += rng.randn(n * m, rank) * 0.01
    V += rng.randn(m * p, rank) * 0.01
    W += rng.randn(p * n, rank) * 0.01
    
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    for it in range(max_iter):
        if _remaining() < 5:
            break
        
        # Update U
        U_new = U.copy()
        for (i, j, k) in indices:
            ui, vi, wi = i * m + j, j * p + k, k * n + i
            vec = V[vi, :] * W[wi, :]
            if np.linalg.norm(vec) < 1e-12:
                continue
            lam = 1e-8
            A = np.outer(vec, vec) + lam * np.eye(rank)
            b = vec
            U_new[ui, :] = np.linalg.solve(A, b)
        U = U_new
        
        # Update V
        V_new = V.copy()
        for (i, j, k) in indices:
            ui, vi, wi = i * m + j, j * p + k, k * n + i
            vec = U[ui, :] * W[wi, :]
            if np.linalg.norm(vec) < 1e-12:
                continue
            lam = 1e-8
            A = np.outer(vec, vec) + lam * np.eye(rank)
            b = vec
            V_new[vi, :] = np.linalg.solve(A, b)
        V = V_new
        
        # Update W
        W_new = W.copy()
        for (i, j, k) in indices:
            ui, vi, wi = i * m + j, j * p + k, k * n + i
            vec = U[ui, :] * V[vi, :]
            if np.linalg.norm(vec) < 1e-12:
                continue
            lam = 1e-8
            A = np.outer(vec, vec) + lam * np.eye(rank)
            b = vec
            W_new[wi, :] = np.linalg.solve(A, b)
        W = W_new
        
        # Check convergence
        recon = np.zeros((n, m, p))
        for c in range(rank):
            u = U[:, c]
            v = V[:, c]
            w = W[:, c]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        recon[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
        err = np.max(np.abs(recon - 1.0))
        if err < tol:
            break
    
    # Final check
    recon = np.zeros((n, m, p))
    for c in range(rank):
        u = U[:, c]
        v = V[:, c]
        w = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    err = np.max(np.abs(recon - 1.0))
    if err < 1e-10:
        return U, V, W
    return None

def _gradient_polish(U, V, W, rank, n, m, p, max_iter=200, lr=0.01):
    """Gradient descent polish to refine the decomposition."""
    T = np.ones((n, m, p))
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iter):
        if _remaining() < 5:
            break
        
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        residual = T - result
        err = np.linalg.norm(residual)
        
        if err < 1e-12:
            break
        
        grad_U = np.einsum('ijk,jkr,kir->ijr', residual, V_r, W_r)
        grad_V = np.einsum('ijk,ijr,kir->jkr', residual, U_r, W_r)
        grad_W = np.einsum('ijk,ijr,jkr->kir', residual, U_r, V_r)
        
        U -= lr * grad_U.reshape(n * m, rank)
        V -= lr * grad_V.reshape(m * p, rank)
        W -= lr * grad_W.reshape(p * n, rank)
    
    recon = np.zeros((n, m, p))
    for c in range(rank):
        u = U[:, c]
        v = V[:, c]
        w = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    err = np.max(np.abs(recon - 1.0))
    if err < 1e-10:
        return U, V, W
    return None

def _try_rank(rank, n, m, p, max_restarts=30):
    """Try to find a rank-r decomposition with multiple restarts."""
    for seed in range(max_restarts):
        if _remaining() < 10:
            break
        result = _als_decompose(rank, n, m, p, max_iter=300, seed=seed)
        if result is not None:
            U, V, W = result
            # Try gradient polish
            polished = _gradient_polish(U, V, W, rank, n, m, p, max_iter=100, lr=0.005)
            if polished is not None:
                U, V, W = polished
            if _verify(U, V, W):
                return U, V, W
    return None

def _schoolbook():
    """Standard schoolbook decomposition, rank = n*m*p = 18."""
    n, m, p = DIMENSIONS
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

def construct():
    """
    Return the rank-minimal exact tensor decomposition for 2x3 * 3x3.
    Search for lower-rank decompositions using ALS with gradient polish.
    """
    n, m, p = DIMENSIONS
    
    # Try ranks from 12 upward (theoretical lower bound is around 12)
    for rank in range(12, 19):
        if _remaining() < 15:
            break
        result = _try_rank(rank, n, m, p, max_restarts=25)
        if result is not None:
            U, V, W = result
            # Convert to list format
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return _schoolbook()
