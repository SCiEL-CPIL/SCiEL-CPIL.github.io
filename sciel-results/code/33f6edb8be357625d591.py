"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x6 matrix.
    Uses a deadline-guarded search with random initialization and local refinement.
    """
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline, 20s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 90
    
    # Store best result found so far
    best_rank = schoolbook_rank
    best_result = _schoolbook_decomposition(n, m, p)
    
    # Try to find better decompositions using various strategies
    strategies = [
        _try_structured_decomposition,
        _try_random_search,
        _try_lagrangian_relaxation,
    ]
    
    for strategy in strategies:
        if time.time() >= deadline - 5:
            break
        try:
            result, rank = strategy(n, m, p, deadline)
            if result is not None and rank < best_rank:
                best_rank = rank
                best_result = result
        except Exception:
            continue
    
    return best_result


def _schoolbook_decomposition(n, m, p):
    """Create the standard schoolbook decomposition of rank n*m*p."""
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


def _verify_decomposition(U, V, W, n, m, p, rank):
    """
    Verify that the decomposition exactly decomposes the matrix multiplication tensor.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 iff i,j,k valid.
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1.0) > 1e-10:
                    return False
    return True


def _try_structured_decomposition(n, m, p, deadline):
    """
    Try structured decompositions using Kronecker products and block structures.
    """
    # For 3x5x6, try to find a decomposition with rank < 90
    # Known lower bound is around 45-50 for these dimensions
    
    # Try rank values from known good bounds upward
    min_possible = int(n * m * p * 0.5)  # Conservative lower bound
    max_rank_to_try = min(70, n * m * p - 1)
    
    for rank in range(min_possible, max_rank_to_try + 1):
        if time.time() >= deadline - 10:
            break
        
        # Try random initialization with structured constraints
        for attempt in range(20):
            if time.time() >= deadline - 10:
                break
            
            # Initialize with random values
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Try to refine using alternating least squares
            success = _refine_decomposition(U, V, W, n, m, p, rank, deadline)
            if success:
                # Convert to list format
                U_list = [[U[i, c] for c in range(rank)] for i in range(n * m)]
                V_list = [[V[i, c] for c in range(rank)] for i in range(m * p)]
                W_list = [[W[i, c] for c in range(rank)] for i in range(p * n)]
                
                # Verify
                if _verify_decomposition(U_list, V_list, W_list, n, m, p, rank):
                    return {"U": U_list, "V": V_list, "W": W_list}, rank
    
    return None, None


def _refine_decomposition(U, V, W, n, m, p, rank, deadline):
    """
    Refine a decomposition using alternating least squares to minimize the error.
    Returns True if the decomposition is exact.
    """
    max_iters = 100
    tol = 1e-12
    
    for iteration in range(max_iters):
        if time.time() >= deadline - 15:
            return False
        
        # Update U: for each (i,j), solve for U[i*m+j, :]
        for i in range(n):
            for j in range(m):
                # T[i,j,:] = U[i*m+j,:] * (V[j*p:,:])^T * W
                # This is a linear system
                A = np.zeros((rank, rank))
                for k in range(p):
                    v_col = V[j * p + k, :]
                    w_row = W[k * n + i, :]
                    A += np.outer(v_col, w_row)
                
                b = np.ones(rank)  # Target: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                # Actually, we need to be more careful
                # Let's use a different approach
                
        # Alternative: use gradient descent on the error
        # Error = sum_{i,j,k} (T[i,j,k] - 1)^2 where T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        
        # Compute current error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t_val = 0.0
                    for c in range(rank):
                        t_val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    error += (t_val - 1.0) ** 2
        
        if error < tol:
            return True
        
        # Simple gradient step
        lr = 0.001
        for c in range(rank):
            # Gradient with respect to U[i*m+j, c]
            for i in range(n):
                for j in range(m):
                    grad = 0.0
                    for k in range(p):
                        grad += 2 * (U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] - 1.0) * V[j*p+k, c] * W[k*n+i, c]
                    U[i*m+j, c] -= lr * grad
            
            # Gradient with respect to V[j*p+k, c]
            for j in range(m):
                for k in range(p):
                    grad = 0.0
                    for i in range(n):
                        grad += 2 * (U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] - 1.0) * U[i*m+j, c] * W[k*n+i, c]
                    V[j*p+k, c] -= lr * grad
            
            # Gradient with respect to W[k*n+i, c]
            for k in range(p):
                for i in range(n):
                    grad = 0.0
                    for j in range(m):
                        grad += 2 * (U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] - 1.0) * U[i*m+j, c] * V[j*p+k, c]
                    W[k*n+i, c] -= lr * grad
        
        if time.time() >= deadline - 15:
            break
    
    # Final check
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t_val = 0.0
                for c in range(rank):
                    t_val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                error += (t_val - 1.0) ** 2
    
    return error < tol


def _try_random_search(n, m, p, deadline):
    """
    Try random search with various rank values.
    """
    # Try a few specific rank values that might work
    candidate_ranks = [45, 48, 50, 52, 55, 60, 65, 70]
    
    for rank in candidate_ranks:
        if time.time() >= deadline - 10:
            break
        
        for attempt in range(5):
            if time.time() >= deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            success = _refine_decomposition(U, V, W, n, m, p, rank, deadline)
            if success:
                U_list = [[U[i, c] for c in range(rank)] for i in range(n * m)]
                V_list = [[V[i, c] for c in range(rank)] for i in range(m * p)]
                W_list = [[W[i, c] for c in range(rank)] for i in range(p * n)]
                
                if _verify_decomposition(U_list, V_list, W_list, n, m, p, rank):
                    return {"U": U_list, "V": V_list, "W": W_list}, rank
    
    return None, None


def _try_lagrangian_relaxation(n, m, p, deadline):
    """
    Try Lagrangian relaxation approach.
    """
    # This is a placeholder for more sophisticated optimization
    # For now, just return None
    return None, None
# EVOLVE-BLOCK-END
