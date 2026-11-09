"""Matrix multiplication 3,4,8 rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 8]

import numpy as np
from itertools import product
import time


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # safety margin before 150s limit

    # Strategy:
    # 1. Try to find a low-rank decomposition using ALS with multiple random restarts
    # 2. The tensor T[i,j,k] = 1 for all i,j,k
    # 3. We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # 4. Start from lower bounds and increase rank until we find a valid decomposition
    # 5. For matrix multiplication, known bounds suggest rank < n*m*p = 96
    
    # Lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # But Strassen-style improvements can reduce this
    lb = max(n * m, m * p, p * n)  # 32
    
    best_rank = n * m * p  # schoolbook = 96
    best_factors = None

    # Try ranks from lower bound up to schoolbook
    for R in range(lb, n * m * p + 1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random restarts for this rank
        success = _als_decompose(n, m, p, R, time_budget=2.0, n_restarts=3)
        if success is not None:
            best_rank = R
            best_factors = success
            break  # found minimal rank, stop
    
    if best_factors is not None:
        U, V, W = best_factors
        # Convert to lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
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


def _als_decompose(n, m, p, R, time_budget=2.0, n_restarts=3):
    """
    Try to find a rank-R decomposition using ALS with multiple restarts.
    Returns (U, V, W) as numpy arrays if successful, else None.
    """
    start_time = time.time()
    
    # Target tensor: T[i,j,k] = 1
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Initialize with random values (different seed for each restart)
        np.random.seed(42 + restart)
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        max_iters = 300
        
        for it in range(max_iters):
            if time.time() - start_time > time_budget:
                break
            
            # Solve for U: for each (i,j), U[i*m+j, :] = argmin_U sum_k [T[i,j,k] - sum_r V[j*p+k,r]*W[k*n+i,r]*U[i*m+j,r]]^2
            for i in range(n):
                for j in range(m):
                    # Build system: for each k, sum_r V[j*p+k, r] * W[k*n+i, r] * U[i*m+j, r] = 1
                    # Let A[r, r'] = sum_k V[j*p+k, r] * W[k*n+i, r'] (R x R matrix)
                    # Then A @ U[i*m+j, :] = 1 (R-vector of ones)
                    A = np.zeros((R, R))
                    for k in range(p):
                        v_row = V[j * p + k, :]  # shape (R,)
                        w_row = W[k * n + i, :]  # shape (R,)
                        A += np.outer(v_row, w_row)
                    # Solve A @ u = ones(R)
                    try:
                        U[i * m + j, :] = np.linalg.solve(A, np.ones(R))
                    except np.linalg.LinAlgError:
                        # Use pseudoinverse
                        U[i * m + j, :] = np.linalg.lstsq(A, np.ones(R), rcond=None)[0]
            
            # Solve for V: for each (j,k), similar
            for j in range(m):
                for k in range(p):
                    A = np.zeros((R, R))
                    for i in range(n):
                        u_row = U[i * m + j, :]  # shape (R,)
                        w_row = W[k * n + i, :]  # shape (R,)
                        A += np.outer(u_row, w_row)
                    try:
                        V[j * p + k, :] = np.linalg.solve(A, np.ones(R))
                    except np.linalg.LinAlgError:
                        V[j * p + k, :] = np.linalg.lstsq(A, np.ones(R), rcond=None)[0]
            
            # Solve for W: for each (k,i), similar
            for k in range(p):
                for i in range(n):
                    A = np.zeros((R, R))
                    for j in range(m):
                        u_row = U[i * m + j, :]  # shape (R,)
                        v_row = V[j * p + k, :]  # shape (R,)
                        A += np.outer(u_row, v_row)
                    try:
                        W[k * n + i, :] = np.linalg.solve(A, np.ones(R))
                    except np.linalg.LinAlgError:
                        W[k * n + i, :] = np.linalg.lstsq(A, np.ones(R), rcond=None)[0]
            
            # Check convergence
            error = _compute_error(n, m, p, U, V, W)
            if error < 1e-8:
                return (U, V, W)
        
        # Check final error for this restart
        error = _compute_error(n, m, p, U, V, W)
        if error < 1e-6:
            return (U, V, W)
    
    return None


def _compute_error(n, m, p, U, V, W):
    """Compute the decomposition error."""
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                error += (val - 1.0) ** 2
    return error
# EVOLVE-BLOCK-END
