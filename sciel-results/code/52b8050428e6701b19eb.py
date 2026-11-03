"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _schoolbook_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix multiplication: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 if this term contributes, but actually the tensor is:
    # T[i,j,k] = delta_{i,k} * delta_{j,j} ... no.
    # 
    # The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k in appropriate ranges
    # when we view it as: output[i,k] = sum_j input1[i,j] * input2[j,k]
    # The tensor is T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 (it's the identity in the sense that each triple contributes)
    # 
    # Actually, the tensor for matrix mult of (n x m) by (m x p) is:
    # T[i, j, k] = 1 for all valid i, j, k
    # because output[i,k] = sum_j A[i,j] B[j,k]
    # 
    # The decomposition requires: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(actual - expected) > tol:
                    return False
    return True


def _try_random_decomposition(n, m, p, rank, seed, deadline, max_iters=500):
    """Try to find a rank-`rank` decomposition using random initialization and gradient-like refinement."""
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # Scale to get roughly correct magnitudes
    U /= np.sqrt(rank)
    V /= np.sqrt(rank)
    W /= np.sqrt(rank)
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.01
    for iteration in range(max_iters):
        if time.time() > deadline - 2.0:
            break
        
        # Compute loss: sum over (i,j,k) of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    err = val - 1.0
                    loss += err * err
                    for c in range(rank):
                        grad_U[i * m + j, c] += 2 * err * V[j * p + k, c] * W[k * n + i, c]
                        grad_V[j * p + k, c] += 2 * err * U[i * m + j, c] * W[k * n + i, c]
                        grad_W[k * n + i, c] += 2 * err * U[i * m + j, c] * V[j * p + k, c]
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Gradient step
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Check if we've converged
        if best_loss < 1e-12:
            break
        
        # Decay learning rate
        lr *= 0.995
    
    if best_loss < 1e-9:
        # Convert to list of lists with rounded values
        U_list = [[round(float(x), 6) for x in row] for row in best_U]
        V_list = [[round(float(x), 6) for x in row] for row in best_V]
        W_list = [[round(float(x), 6) for x in row] for row in best_W]
        
        # Verify
        if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-6):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Try ranks from known lower bounds up to schoolbook
    # For 2x3x6, schoolbook is 36. Known lower bounds suggest rank could be lower.
    # Try a range of ranks.
    min_rank = max(n * m, m * p, p * n)  # Simple lower bound
    max_rank = n * m * p
    
    best_result = None
    
    for rank in range(min_rank, max_rank + 1):
        if time.time() > deadline - 5.0:
            break
        
        # Try multiple random seeds for this rank
        for seed in range(10):
            if time.time() > deadline - 3.0:
                break
            
            result = _try_random_decomposition(n, m, p, rank, seed, deadline, max_iters=200)
            if result is not None:
                if best_result is None or rank < len(best_result["U"][0]):
                    best_result = result
                # Found a valid decomposition at this rank, try to go lower
                break
        
        # If we found a valid one, we could stop, but let's see if we can go lower
        # Actually, we're iterating rank upward, so first valid is the lowest
        if best_result is not None:
            break
    
    if best_result is None:
        # Fall back to schoolbook
        return _schoolbook_decomposition(n, m, p)
    
    return best_result
# EVOLVE-BLOCK-END
