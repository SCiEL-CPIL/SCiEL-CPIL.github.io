"""
Tensor decomposition for 3x3x8 matrix multiplication.
Uses alternating least-squares (ALS) with multiple random restarts to find
a rank-minimal exact decomposition. Falls back to schoolbook rank 72.
"""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]
DEADLINE = 120.0

# EVOLVE-BLOCK-START

def _build_schoolbook():
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

def _verify(U, V, W, atol=1e-6):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    T = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        U_r = Ua[:, r]
        V_r = Va[:, r]
        W_r = Wa[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U_r[i * m + j] * V_r[j * p + k] * W_r[k * n + i]
    return np.allclose(T, 1.0, atol=atol)

def _compute_loss(U, V, W):
    """Compute the sum-of-squares loss for the tensor decomposition."""
    n, m, p = DIMENSIONS
    total = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                total += (val - 1.0) ** 2
    return total

def _solve_for_W(U, V, r):
    """Solve for W given U and V using least squares."""
    n, m, p = DIMENSIONS
    W = np.zeros((p * n, r))
    for k in range(p):
        for i in range(n):
            A = np.zeros((m, r))
            for j in range(m):
                A[j, :] = U[i * m + j, :] * V[j * p + k, :]
            try:
                W_col, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                W[k * n + i, :] = W_col
            except:
                W[k * n + i, :] = 0
    return W

def _solve_for_U(U, V, W, r):
    """Solve for U given V and W using least squares."""
    n, m, p = DIMENSIONS
    U_new = np.zeros((n * m, r))
    for i in range(n):
        for j in range(m):
            A = np.zeros((p, r))
            for k in range(p):
                A[k, :] = V[j * p + k, :] * W[k * n + i, :]
            try:
                U_col, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                U_new[i * m + j, :] = U_col
            except:
                U_new[i * m + j, :] = 0
    return U_new

def _solve_for_V(U, V, W, r):
    """Solve for V given U and W using least squares."""
    n, m, p = DIMENSIONS
    V_new = np.zeros((m * p, r))
    for j in range(m):
        for k in range(p):
            A = np.zeros((n, r))
            for i in range(n):
                A[i, :] = U[i * m + j, :] * W[k * n + i, :]
            try:
                V_col, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                V_new[j * p + k, :] = V_col
            except:
                V_new[j * p + k, :] = 0
    return V_new

def _try_rank_als(r, max_sweeps=200, n_restarts=5, seed=42):
    """
    Try to find a rank-r decomposition using alternating least squares.
    Returns (U, V, W) as numpy arrays if successful, else None.
    """
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    for restart in range(n_restarts):
        # Initialize with random values
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for sweep in range(max_sweeps):
            # Alternate solving for each factor
            W = _solve_for_W(U, V, r)
            U = _solve_for_U(U, V, W, r)
            V = _solve_for_V(U, V, W, r)
            
            loss = _compute_loss(U, V, W)
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-12:
                break
            
            # Check deadline
            if time.time() > DEADLINE - 15:
                return None
        
        if best_loss < 1e-10:
            # Try to round to half-integers for cleaner output
            U_r = np.round(best_U * 2) / 2
            V_r = np.round(best_V * 2) / 2
            W_r = np.round(best_W * 2) / 2
            
            # Verify rounded version
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_r[i * m + j], V_r[j * p + k] * W_r[k * n + i])
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-6:
                return U_r, V_r, W_r
            else:
                # Return unrounded if it's accurate
                return best_U, best_V, best_W
    
    return None

def construct():
    start_time = time.time()
    
    # Build schoolbook as fallback
    schoolbook = _build_schoolbook()
    
    # Try ranks from 71 down to 40
    for r in range(71, 39, -1):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        result = _try_rank_als(r, max_sweeps=150, n_restarts=3, seed=42 + r)
        if result is not None:
            U, V, W = result
            return {
                "U": [[float(x) for x in row] for row in U],
                "V": [[float(x) for x in row] for row in V],
                "W": [[float(x) for x in row] for row in W]
            }
    
    return schoolbook
# EVOLVE-BLOCK-END
