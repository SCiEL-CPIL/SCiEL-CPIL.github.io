"""Program for matrix multiplication tensor decomposition 2x3 by 3x7."""
DIMENSIONS = [2, 3, 7]

import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    rank = len(U[0])
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                T[i, j, k] = val
    return np.allclose(T, 1.0)

def _try_rank_reduction(n, m, p, target_rank, time_budget):
    """
    Attempt to find a rank-target_rank decomposition using polynomial structure.
    Uses the fact that m=3 allows representing the middle dimension as polynomials.
    Returns (U, V, W) if found, None otherwise.
    """
    start_time = time.time()
    
    # Initialize with random matrices
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1
    
    # Define the tensor T[i,j,k] = 1 for all i,j,k
    # We want: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    def compute_error():
        """Compute the Frobenius norm of the error tensor."""
        err = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    err[i, j, k] = 1.0 - np.dot(
                        U[i*m+j, :], 
                        V[j*p+k, :] * W[k*n+i, :]
                    )
        return np.linalg.norm(err)
    
    # Gradient-based optimization
    lr = 0.01
    best_error = compute_error()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for step in range(2000):
        if time.time() - start_time > time_budget:
            break
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    err_val = 1.0 - np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    # d/dU: -err * V[j*p+k,:] * W[k*n+i,:]
                    dU[i*m+j, :] -= err_val * V[j*p+k, :] * W[k*n+i, :]
                    # d/dV: -err * U[i*m+j,:] * W[k*n+i,:]
                    dV[j*p+k, :] -= err_val * U[i*m+j, :] * W[k*n+i, :]
                    # d/dW: -err * U[i*m+j,:] * V[j*p+k,:]
                    dW[k*n+i, :] -= err_val * U[i*m+j, :] * V[j*p+k, :]
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        err = compute_error()
        if err < best_error:
            best_error = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_error < 1e-8:
            break
        
        # Adaptive learning rate
        if err > best_error * 1.5:
            lr *= 0.9
        
    if best_error < 1e-6:
        # Round to half-integers
        best_U = np.round(best_U * 2) / 2
        best_V = np.round(best_V * 2) / 2
        best_W = np.round(best_W * 2) / 2
        
        # Verify after rounding
        if _verify_decomposition(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def _systematic_search(n, m, p, time_budget):
    """
    Systematic search for lower-rank decompositions using polynomial structure.
    Tries various ranks from low to high.
    """
    start_time = time.time()
    schoolbook_rank = n * m * p
    
    # Try ranks from 20 to schoolbook_rank-1
    for target_rank in range(20, schoolbook_rank):
        if time.time() - start_time > time_budget - 5:
            break
        
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time < 10:
            break
        
        # Try multiple random restarts
        for attempt in range(3):
            if time.time() - start_time > time_budget - 5:
                break
            
            result = _try_rank_reduction(n, m, p, target_rank, remaining_time / 3)
            if result is not None:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    """
    Returns a valid tensor decomposition for matrix multiplication 2x3 * 3x7.
    First attempts to find a lower-rank decomposition using polynomial structure,
    falls back to schoolbook if not found within time budget.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_budget = 140  # Leave 10s margin before 150s limit
    
    # First, try to find a lower-rank decomposition
    result = _systematic_search(n, m, p, time_budget)
    
    if result is not None:
        return result
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][idx] = 1
                V[v_idx][idx] = 1
                W[w_idx][idx] = 1
                idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

# Verification helper (not part of the required interface, but useful for local debugging)
def verify_decomposition(U, V, W, n, m, p):
    """
    Verifies that the tensor decomposition is exact.
    T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    rank = len(U[0])
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                T[i, j, k] = val
    return np.allclose(T, 1.0)

if __name__ == "__main__":
    result = construct()
    U, V, W = result["U"], result["V"], result["W"]
    n, m, p = DIMENSIONS
    if verify_decomposition(U, V, W, n, m, p):
        print("Decomposition is valid.")
    else:
        print("Decomposition is INVALID.")
    print(f"Rank: {len(U[0])}")
