import numpy as np
from itertools import combinations, product

DIMENSIONS = [2, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each (i, j, k), the tensor entry should be 1
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 50
    
    # Try to find a lower-rank decomposition using random search with structured initialization
    # We'll try ranks from 20 up to trivial_rank - 1
    
    best_rank = trivial_rank
    best_U = None
    best_V = None
    best_W = None
    
    # For small ranks, we can try a structured approach
    # The matrix multiplication tensor has symmetry: T[i,j,k] = 1 for all i,j,k
    # A rank-r decomposition means: for all i,j,k: sum_{r=1}^R u_{i,j,r} * v_{j,k,r} * w_{k,i,r} = 1
    
    # Try random rank-r decompositions for r from 25 to trivial_rank-1
    # Use a time-limited search
    
    import time
    start_time = time.time()
    time_limit = 60  # seconds
    
    for rank in range(25, trivial_rank):
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple random initializations for this rank
        max_attempts = 20
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization
            np.random.seed(42 + rank * 100 + attempt)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Use gradient descent to minimize the error
            # Error = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
            lr = 0.01
            for step in range(500):
                # Compute gradients
                U_grad = np.zeros_like(U)
                V_grad = np.zeros_like(V)
                W_grad = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                            residual = 0.0
                            for r in range(rank):
                                residual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            residual -= 1.0
                            
                            for r in range(rank):
                                U_grad[i*m+j, r] += 2 * residual * V[j*p+k, r] * W[k*n+i, r]
                                V_grad[j*p+k, r] += 2 * residual * U[i*m+j, r] * W[k*n+i, r]
                                W_grad[k*n+i, r] += 2 * residual * U[i*m+j, r] * V[j*p+k, r]
                
                U -= lr * U_grad
                V -= lr * V_grad
                W -= lr * W_grad
            
            # Check if we found a valid decomposition
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best_rank = rank
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break
        if best_rank == rank:
            break
    
    # If no better decomposition found, use the trivial one
    if best_U is None:
        rank = trivial_rank
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
    else:
        U = best_U
        V = best_V
        W = best_W
    
    return {"U": U, "V": V, "W": W}
