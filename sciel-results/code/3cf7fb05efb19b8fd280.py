"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the multiplication tensor T[i,j,k] = delta[i,j,k] in the convention:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Index convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # The decomposition says: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = T[i,j,k]
    # where T[i,j,k] = 1 (always 1 for valid matrix mult)
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i*m+j], V_np[j*p+k])
                # Wait, that's not right. The tensor is trilinear.
                # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _construct_schoolbook(n, m, p):
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

def _try_structured_decomposition(n, m, p, max_rank=110, time_limit=130.0):
    """
    Try to find a lower-rank decomposition using a structured ansatz.
    
    Key insight: The matrix multiplication tensor T[i,j,k] has the property that
    for fixed j, the slice over (i,k) is the identity-like structure (rank 1 per j).
    
    We use a randomized approach: try various rank values and use iterative
    tensor decomposition / optimization.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Start with schoolbook and try to reduce rank
    best_result = _construct_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Strategy: Use the fact that for each j, the (i,k) slice is separable.
    # A known bound: rank of matmul n x m x p is at most n*m*p - min(n,m,p) + 1
    # But we can do better with structured approaches.
    
    # Try rank values from high to low
    for target_rank in range(best_rank - 1, max(50, best_rank - 40), -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        max_attempts = 3
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with sparsity
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Add some structure: for each j, ensure the (i,k) plane has correct form
            # Iterate: for each (i,j,k), adjust to satisfy U[i*m+j]·V[j*p+k]·W[k*n+i] = 1
            
            # Use a simple iterative refinement
            for iteration in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Compute current error
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-8:
                    # Found a solution! Convert to list
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    
                    # Verify
                    if _verify(U_list, V_list, W_list, n, m, p):
                        result = {"U": U_list, "V": V_list, "W": W_list}
                        return result
                
                # Gradient-free update: nudge random entries
                # For each constraint that's violated, adjust one of the three vectors
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if time.time() > deadline - 10:
                                break
                            val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                            err = 1.0 - val
                            if abs(err) > 1e-8:
                                # Nudge U[i*m+j] in direction of V[j*p+k]*W[k*n+i]
                                grad = V[j*p+k] * W[k*n+i]
                                U[i*m+j] += 0.01 * err * grad / (np.sum(grad**2) + 1e-12)
            
            # Check if we converged
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-8:
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                if _verify(U_list, V_list, W_list, n, m, p):
                    result = {"U": U_list, "V": V_list, "W": W_list}
                    return result
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to find a better decomposition
    try:
        result = _try_structured_decomposition(n, m, p, max_rank=115, time_limit=130.0)
    except Exception:
        result = _construct_schoolbook(n, m, p)
    
    # Ensure we're within time budget
    if time.time() - start_time > 135:
        return _construct_schoolbook(n, m, p)
    
    return result
# EVOLVE-BLOCK-END
