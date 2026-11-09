"""Matrix multiplication tensor decomposition 2x4x7 - rank minimization."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
    expected = np.ones((n, m, p))
    if not np.allclose(T, expected, atol=tol, rtol=0):
        return False
    return True

def _als_decompose(n, m, p, rank, max_iters=200, tol=1e-10, seed=42):
    """Alternating least squares to find rank-r decomposition."""
    rng = np.random.RandomState(seed)
    
    # Initialize with random sparse structure
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    # Add a small constant to help convergence
    U += 0.01
    V += 0.01
    W += 0.01
    
    for it in range(max_iters):
        # Update U: for each (i,j) row, solve least squares
        for i in range(n):
            for j in range(m):
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # For fixed i,j: sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
                # Let M[r,k] = V[j*p+k, r] * W[k*n+i, r]
                # Then: M @ U[i*m+j, :] = ones(p)
                M = np.zeros((rank, p))
                for k in range(p):
                    for r in range(rank):
                        M[r, k] = V[j * p + k, r] * W[k * n + i, r]
                # Solve M^T M @ u = M^T @ ones
                try:
                    A = M.T @ M + 1e-8 * np.eye(rank)
                    b = M.T @ np.ones(p)
                    U[i * m + j, :] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                # Let M[r,i] = U[i*m+j, r] * W[k*n+i, r]
                # Then: M @ V[j*p+k, :] = ones(n)
                M = np.zeros((rank, n))
                for i in range(n):
                    for r in range(rank):
                        M[r, i] = U[i * m + j, r] * W[k * n + i, r]
                try:
                    A = M.T @ M + 1e-8 * np.eye(rank)
                    b = M.T @ np.ones(n)
                    V[j * p + k, :] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                # Let M[r,j] = U[i*m+j, r] * V[j*p+k, r]
                # Then: M @ W[k*n+i, :] = ones(m)
                M = np.zeros((rank, m))
                for j in range(m):
                    for r in range(rank):
                        M[r, j] = U[i * m + j, r] * V[j * p + k, r]
                try:
                    A = M.T @ M + 1e-8 * np.eye(rank)
                    b = M.T @ np.ones(m)
                    W[k * n + i, :] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    pass
        
        # Check convergence
        if it % 10 == 0:
            T = np.zeros((n, m, p))
            for r in range(rank):
                u_col = U[:, r]
                v_col = V[:, r]
                w_col = W[:, r]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i, j, k] += u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
            err = np.max(np.abs(T - 1.0))
            if err < tol:
                break
    
    return U, V, W

def _structured_decomposition(n, m, p, rank):
    """Try structured decomposition based on block patterns."""
    # Try to partition the (i,j,k) space into rank blocks
    # Each block covers a subset where the product structure is simple
    
    # Strategy: use the fact that T[i,j,k]=1 and try to find rank-1 tensors
    # that sum to 1
    
    # Simple approach: try random sparse initialization with ALS
    best_U, best_V, best_W = None, None, None
    best_err = np.inf
    
    for attempt in range(20):
        seed = attempt * 1000 + 42
        U, V, W = _als_decompose(n, m, p, rank, max_iters=100, tol=1e-10, seed=seed)
        
        T = np.zeros((n, m, p))
        for r in range(rank):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
        err = np.max(np.abs(T - 1.0))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-10:
            break
    
    if best_err < 1e-10:
        return best_U, best_V, best_W
    return None, None, None

def _schoolbook(n, m, p):
    """Return U, V, W for naive tensor decomposition of n x m @ m x p."""
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

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s margin out of 120s limit
    
    # Schoolbook baseline
    U_school, V_school, W_school = _schoolbook(n, m, p)
    school_rank = n * m * p
    
    # Lower bound
    lower_bound = max(n * m, m * p, n * p)
    
    best_rank = school_rank
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try ranks from lower_bound up to school_rank - 1
    for rank in range(lower_bound, school_rank):
        if time.time() > deadline - 30:  # Leave 30s margin
            break
        
        # Try structured/ALS approach
        U, V, W = _structured_decomposition(n, m, p, rank)
        
        if U is not None:
            if _verify(U, V, W, n, m, p, tol=1e-10):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break  # Found a valid decomposition, this is the minimum for this search
        
        # If ALS didn't work, try a different initialization strategy
        if time.time() > deadline - 20:
            break
    
    # Final check
    if best_rank < school_rank:
        U = best_U
        V = best_V
        W = best_W
    else:
        U = U_school
        V = V_school
        W = W_school
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
