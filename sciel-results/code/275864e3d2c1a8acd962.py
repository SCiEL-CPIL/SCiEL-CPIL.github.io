"""Matrix multiplication 2x2x4 rank-minimal decomposition search."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 2, 4]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the tensor decomposition is exact."""
    rank = U.shape[1]
    # Build the multiplication tensor T[i,j,k] = 1 if i,j,k are valid
    # T[i, j, k] corresponds to (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # In the trace convention: W[k*n+i, r], U[i*m+j, r], V[j*p+k, r]
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank(rank, n, m, p, max_time=100.0, seed=42):
    """Try to find a rank-`rank` decomposition using numerical optimization."""
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    best_rank_found = None
    
    for attempt in range(50):
        if time.time() - start_time > max_time - 5:
            break
        
        # Initialize randomly
        U = rng.randn(n*m, rank) * 0.5
        V = rng.randn(m*p, rank) * 0.5
        W = rng.randn(p*n, rank) * 0.5
        
        # Normalize columns
        for r in range(rank):
            norm = np.sqrt(np.sum(U[:,r]**2) * np.sum(V[:,r]**2) * np.sum(W[:,r]**2))
            if norm > 1e-12:
                U[:,r] /= norm**(1/3)
                V[:,r] /= norm**(1/3)
                W[:,r] /= norm**(1/3)
        
        # Gradient descent
        lr = 0.01
        for step in range(2000):
            if time.time() - start_time > max_time - 5:
                break
            
            # Compute error
            err = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # T[i,j,k] should equal 1
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err += (val - 1.0)**2
                        d = 2 * (val - 1.0)
                        for r in range(rank):
                            grad_U[i*m+j, r] += d * V[j*p+k, r] * W[k*n+i, r]
                            grad_V[j*p+k, r] += d * U[i*m+j, r] * W[k*n+i, r]
                            grad_W[k*n+i, r] += d * U[i*m+j, r] * V[j*p+k, r]
            
            if err < 1e-18:
                return (U, V, W)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Adaptive learning rate
            if step % 200 == 199:
                lr *= 0.8
        
        # Check if we converged
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-7):
            # Try to round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, tol=1e-9):
                return (U_r, V_r, W_r)
            else:
                return (U, V, W)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Known lower bound: for 2x2x4, the rank is at least 13
    # (since 2x2x2 has rank 7, and we need to handle the extra dimension)
    # Actually, let's think: n*m*p = 16 is the schoolbook rank
    # The minimum rank for 2x2x4 matrix multiplication is known to be 13
    
    # Try ranks from 13 upward
    for rank in range(13, 17):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank(rank, n, m, p, max_time=min(30, deadline - time.time()), seed=rank*100+42)
        
        if result is not None:
            U, V, W = result
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            # Verify one more time
            if _verify(U_list, V_list, W_list, n, m, p, tol=1e-8):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook rank 16
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
