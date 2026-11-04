"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

import time
import sys
import numpy as np
import random

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid exact decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _compute_residual(U, V, W, n, m, p):
    """Compute the Frobenius norm of the residual tensor."""
    residual_sq = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(len(U[0])):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                residual_sq += (s - 1.0)**2
    return np.sqrt(residual_sq)

def _decompose_tensor_vectorized(n, m, p, rank, max_iter=300, tol=1e-10, seed=42, init_method='random'):
    """
    Find a rank decomposition using alternating least squares (ALS).
    
    We seek U (n*m x rank), V (m*p x rank), W (p*n x rank) such that:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rng = np.random.default_rng(seed)
    
    if init_method == 'random':
        U = rng.standard_normal((n*m, rank))
        V = rng.standard_normal((m*p, rank))
        W = rng.standard_normal((p*n, rank))
    elif init_method == 'structured':
        # Use structured initialization based on the tensor structure
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for r in range(rank):
            i = r % n
            j = (r // n) % m
            k = (r // (n*m)) % p
            U[i*m+j, r] = 1.0
            V[j*p+k, r] = 1.0
            W[k*n+i, r] = 1.0
            # Add random perturbation
            U[i*m+j, r] += 0.1 * rng.standard_normal()
            V[j*p+k, r] += 0.1 * rng.standard_normal()
            W[k*n+i, r] += 0.1 * rng.standard_normal()
    else:
        U = rng.standard_normal((n*m, rank))
        V = rng.standard_normal((m*p, rank))
        W = rng.standard_normal((p*n, rank))
    
    best_residual = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Update U: for each (i,j), solve linear system
        for i in range(n):
            for j in range(m):
                idx = i*m + j
                # Build matrix A where A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                A = np.zeros((p, rank))
                b = np.ones(p)  # T[i,j,k] = 1 for all k
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve A^T A U[idx,:] = A^T b
                ATA = A.T @ A
                ATb = A.T @ b
                try:
                    U[idx, :] = np.linalg.solve(ATA + 1e-10*np.eye(rank), ATb)
                except np.linalg.LinAlgError:
                    U[idx, :] = np.linalg.lstsq(ATA, ATb, rcond=None)[0]
        
        # Update V: for each (j,k), solve linear system
        for j in range(m):
            for k in range(p):
                idx = j*p + k
                A = np.zeros((n, rank))
                b = np.ones(n)
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                ATA = A.T @ A
                ATb = A.T @ b
                try:
                    V[idx, :] = np.linalg.solve(ATA + 1e-10*np.eye(rank), ATb)
                except np.linalg.LinAlgError:
                    V[idx, :] = np.linalg.lstsq(ATA, ATb, rcond=None)[0]
        
        # Update W: for each (k,i), solve linear system
        for k in range(p):
            for i in range(n):
                idx = k*n + i
                A = np.zeros((m, rank))
                b = np.ones(m)
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                ATA = A.T @ A
                ATb = A.T @ b
                try:
                    W[idx, :] = np.linalg.solve(ATA + 1e-10*np.eye(rank), ATb)
                except np.linalg.LinAlgError:
                    W[idx, :] = np.linalg.lstsq(ATA, ATb, rcond=None)[0]
        
        # Compute residual
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    residual += (s - 1.0)**2
        residual = np.sqrt(residual)
        
        if residual < best_residual:
            best_residual = residual
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if residual < tol:
            break
    
    return best_U, best_V, best_W, best_residual

def _snap_to_half_int(arr):
    """Snap a numpy array to the nearest integer or half-integer."""
    # Multiply by 2, round to nearest integer, divide by 2
    return np.round(arr * 2.0) / 2.0

def _is_valid_half_int_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W have integer/half-integer entries and form a valid decomposition."""
    # Check entries
    if not np.allclose(U, _snap_to_half_int(U), atol=1e-9):
        return False
    if not np.allclose(V, _snap_to_half_int(V), atol=1e-9):
        return False
    if not np.allclose(W, _snap_to_half_int(W), atol=1e-9):
        return False
        
    # Check decomposition
    U_s = _snap_to_half_int(U)
    V_s = _snap_to_half_int(V)
    W_s = _snap_to_half_int(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(len(U[0])):
                    s += U_s[i*m+j, r] * V_s[j*p+k, r] * W_s[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank(n, m, p, rank, max_restarts=10, time_limit=15):
    """Try to find a rank decomposition of the given rank with integer/half-integer entries."""
    start_time = time.time()
    
    best_residual = np.inf
    best_result = None
    
    init_methods = ['random', 'structured', 'random', 'random', 'structured']
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        seed = 42 + restart * 137
        init_method = init_methods[restart % len(init_methods)]
        U, V, W, residual = _decompose_tensor_vectorized(n, m, p, rank, max_iter=200, tol=1e-10, seed=seed, init_method=init_method)
        
        if residual < best_residual:
            best_residual = residual
            best_result = (U, V, W)
        
        if residual < 1e-8:
            # Try to snap to half-integers
            U_snap = _snap_to_half_int(U)
            V_snap = _snap_to_half_int(V)
            W_snap = _snap_to_half_int(W)
            
            if _is_valid_half_int_decomposition(U_snap, V_snap, W_snap, n, m, p):
                U_list = [[float(x) for x in row] for row in U_snap]
                V_list = [[float(x) for x in row] for row in V_snap]
                W_list = [[float(x) for x in row] for row in W_snap]
                return {"U": U_list, "V": V_list, "W": W_list}
            break
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # Try ranks from lower bound up to schoolbook
    lower_bound = max(n*m, m*p, p*n)
    
    # Try ranks from lower_bound up to schoolbook-1
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline - 5:
            break
        
        result = _try_rank(n, m, p, rank, max_restarts=8, time_limit=12)
        if result is not None:
            best = result
            best_rank = rank
            break  # Found a valid decomposition, this is the minimum
    
    return best
# EVOLVE-BLOCK-END
