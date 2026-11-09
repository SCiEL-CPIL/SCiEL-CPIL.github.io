"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

import numpy as np
from itertools import product
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k
    # We want to find the minimal rank R such that T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r[i,j], V_r[j,k], W_r[k,i] (with W index convention W[k*n+i])
    
    # Strategy: Try rank from theoretical lower bound up to 96
    # Lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # But for matrix multiplication, Strassen's algorithm suggests better bounds
    # For 3x4x4, Strassen gives rank 21; we can try to extend ideas
    
    # Let's try to find a decomposition by solving a linear system
    # For a given rank R, we have R * (n*m + m*p + p*n) variables
    # and n*m*p equations (one for each (i,j,k))
    # T[i,j,k] = sum_r U[r][i*m+j] * V[r][j*p+k] * W[r][k*n+i] = 1
    
    # This is a nonlinear system. Instead, let's use a different approach:
    # Fix U and V, then solve for W linearly, or use iterative methods.
    
    # Alternative: Use the fact that the tensor is separable in a structured way.
    # We can try random initializations and use gradient descent or ALS.
    
    # Let's try a direct approach: since the tensor is "all ones", 
    # we can try to find low-rank factors by exploiting symmetry.
    
    # For now, let's use a greedy/constructive approach.
    # The key insight: if we can partition the index set into groups
    # where within each group, the contribution is a rank-1 tensor,
    # we can reduce the rank.
    
    # Actually, let me try to use numpy to solve this.
    # For a fixed rank R, we can set up the problem as:
    # Find U (n*m x R), V (m*p x R), W (p*n x R)
    # such that for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a tensor decomposition problem. Let's use ALS (Alternating Least Squares).
    
    best_rank = n * m * p  # schoolbook
    best_factors = None
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound for matrix multiplication is at least max(n*m, m*p, p*n)
    lb = max(n * m, m * p, p * n)
    
    for R in range(lb, n * m * p + 1):
        if time.time() > deadline - 10:
            break
        
        # Try to find a rank-R decomposition using ALS
        success = _als_decompose(n, m, p, R, time_budget=2.0)
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


def _als_decompose(n, m, p, R, time_budget=2.0):
    """
    Try to find a rank-R decomposition using ALS.
    Returns (U, V, W) as numpy arrays if successful, else None.
    """
    start_time = time.time()
    
    # Initialize with random values
    np.random.seed(42)
    U = np.random.randn(n * m, R) * 0.1
    V = np.random.randn(m * p, R) * 0.1
    W = np.random.randn(p * n, R) * 0.1
    
    # Target tensor: T[i,j,k] = 1
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # ALS: alternate between solving for U, V, W given the others fixed
    
    max_iters = 200
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Solve for U: for each (i,j), U[i*m+j, :] = argmin_U sum_k [T[i,j,k] - sum_r V[j*p+k,r]*W[k*n+i,r]*U[i*m+j,r]]^2
        # This is a linear least squares problem for each (i,j)
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
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    error += (val - 1.0) ** 2
        if error < 1e-6:
            return (U, V, W)
    
    # Check final error
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                error += (val - 1.0) ** 2
    
    if error < 1e-4:
        return (U, V, W)
    
    return None
# EVOLVE-BLOCK-END
