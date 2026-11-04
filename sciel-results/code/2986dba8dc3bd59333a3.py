"""Task: rank-minimal exact tensor decomposition of 3x4 by 4x7 matrix multiplication.
Trace convention: out[i,k] = sum_j A[i,j]*B[j,k] == sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
"""
import numpy as np
import time
import itertools

_START = time.time()
_LIMIT = 140.0  # safety margin before 150s wall-clock kill
DIMENSIONS = [3, 4, 7]


def _check_decomposition(U, V, W, n, m, p, r, tol=1e-8):
    """Check if the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for rr in range(r):
                    s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _check_fast(U_np, V_np, W_np, n, m, p, r, tol=1e-8):
    """Fast numpy check."""
    for j in range(m):
        u_col = U_np[np.arange(n)*m + j, :]  # (n, r)
        v_col = V_np[j*p + np.arange(p), :]   # (p, r)
        W_reshaped = W_np.reshape(p, n, r)
        W_t = W_reshaped.transpose(1, 0, 2)  # (n, p, r)
        prod = u_col[:, np.newaxis, :] * v_col[np.newaxis, :, :] * W_t  # (n, p, r)
        sums = prod.sum(axis=2)  # (n, p)
        if np.any(np.abs(sums - 1.0) > tol):
            return False
    return True


def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank


def _als_decompose(n, m, p, r, max_iter=5000, tol=1e-10, seed=0):
    """Alternating least squares for tensor decomposition.
    
    For each (i,j): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
    For each (j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
    For each (k,i): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
    """
    np.random.seed(seed)
    
    # Initialize
    U = np.random.randn(n*m, r) * 0.1 + 0.5
    V = np.random.randn(m*p, r) * 0.1 + 0.5
    W = np.random.randn(p*n, r) * 0.1 + 0.5
    
    # Precompute index mappings
    # For U update: for each (i,j), solve U[i*m+j,:] from:
    #   sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
    #   => U[i*m+j,:] @ M_ij = ones(p) where M_ij[k,r] = V[j*p+k,r]*W[k*n+i,r]
    
    # For V update: for each (j,k), solve V[j*p+k,:] from:
    #   sum_r V[j*p+k,r] * (U[i*m+j,r] * W[k*n+i,r]) = 1 for all i
    #   => V[j*p+k,:] @ N_jk = ones(n) where N_jk[i,r] = U[i*m+j,r]*W[k*n+i,r]
    
    # For W update: for each (k,i), solve W[k*n+i,:] from:
    #   sum_r W[k*n+i,r] * (U[i*m+j,r] * V[j*p+k,r]) = 1 for all j
    #   => W[k*n+i,:] @ P_ki = ones(m) where P_ki[j,r] = U[i*m+j,r]*V[j*p+k,r]
    
    prev_err = float('inf')
    
    for it in range(max_iter):
        # Update U
        for i in range(n):
            for j in range(m):
                # M[k,r] = V[j*p+k,r] * W[k*n+i,r]
                M = np.zeros((p, r))
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve: U[i*m+j,:] @ M = ones(p)
                # i.e., M.T @ U[i*m+j,:] = ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                except np.linalg.LinAlgError:
                    U[i*m+j, :] = np.random.randn(r) * 0.1 + 0.5
        
        # Update V
        for j in range(m):
            for k in range(p):
                # N[i,r] = U[i*m+j,r] * W[k*n+i,r]
                N = np.zeros((n, r))
                for i in range(n):
                    N[i, :] = U[i*m+j, :] * W[k*n+i, :]
                # Solve: V[j*p+k,:] @ N = ones(n)
                # i.e., N.T @ V[j*p+k,:] = ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(N.T, np.ones(n), rcond=None)[0]
                except np.linalg.LinAlgError:
                    V[j*p+k, :] = np.random.randn(r) * 0.1 + 0.5
        
        # Update W
        for k in range(p):
            for i in range(n):
                # P[j,r] = U[i*m+j,r] * V[j*p+k,r]
                P = np.zeros((m, r))
                for j in range(m):
                    P[j, :] = U[i*m+j, :] * V[j*p+k, :]
                # Solve: W[k*n+i,:] @ P = ones(m)
                # i.e., P.T @ W[k*n+i,:] = ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(P.T, np.ones(m), rcond=None)[0]
                except np.linalg.LinAlgError:
                    W[k*n+i, :] = np.random.randn(r) * 0.1 + 0.5
        
        # Check convergence
        if it % 10 == 0 or it == max_iter - 1:
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err += (s - 1.0)**2
            if err < tol:
                return U, V, W, True
            if abs(err - prev_err) < tol * 1e-3:
                # Stalled
                if err < 1e-6:
                    return U, V, W, True
                return U, V, W, False
            prev_err = err
    
    # Final check
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err += (s - 1.0)**2
    return U, V, W, err < 1e-6


def _try_als_search(n, m, p, deadline):
    """Try ALS for ranks from lower bound upward."""
    lower_bound = max(n * p, m * p, n * m)  # 28
    
    for r in range(lower_bound, n * m * p + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            U, V, W, success = _als_decompose(n, m, p, r, max_iter=2000, tol=1e-10, seed=r*1000+attempt)
            
            if success and _check_fast(U, V, W, n, m, p, r, tol=1e-8):
                return r, (U.tolist(), V.tolist(), W.tolist())
        
        # Also try with different initialization strategies
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            np.random.seed(50000 + r*1000 + attempt)
            # Try sparse initialization
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            # Initialize with random +/-1 patterns
            U[np.random.randint(0, n*m, size=r), np.arange(r)] = 1
            V[np.random.randint(0, m*p, size=r), np.arange(r)] = 1
            W[np.random.randint(0, p*n, size=r), np.arange(r)] = 1
            
            # Run ALS from this init
            U2, V2, W2, success = _als_decompose(n, m, p, r, max_iter=2000, tol=1e-10, seed=r*2000+attempt)
            
            if success and _check_fast(U2, V2, W2, n, m, p, r, tol=1e-8):
                return r, (U2.tolist(), V2.tolist(), W2.tolist())
    
    return None, None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + _LIMIT - 5.0  # 5s safety margin
    
    # Step 1: Build schoolbook decomposition (always valid)
    U0, V0, W0, R0 = _build_schoolbook(n, m, p)
    
    best_rank = R0
    best_U, best_V, best_W = U0, V0, W0
    
    # Step 2: Try ALS search for lower ranks
    if time.time() < deadline:
        rank_found, decomp_found = _try_als_search(n, m, p, deadline)
        if decomp_found is not None and rank_found < best_rank:
            best_rank = rank_found
            best_U, best_V, best_W = decomp_found
    
    # Ensure we return lists (not numpy arrays)
    if isinstance(best_U, np.ndarray):
        best_U = best_U.tolist()
    if isinstance(best_V, np.ndarray):
        best_V = best_V.tolist()
    if isinstance(best_W, np.ndarray):
        best_W = best_W.tolist()
    
    return {"U": best_U, "V": best_V, "W": best_W}
