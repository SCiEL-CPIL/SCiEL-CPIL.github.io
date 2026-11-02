"""Program for 3x3x6 matrix multiplication tensor decomposition."""
import time
import sys
import numpy as np

DIMENSIONS = [3, 3, 6]

def _schoolbook():
    n, m, p = 3, 3, 6
    rank = n * m * p  # 54
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

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = 3, 3, 6
    rank = len(U[0])
    # Check shapes
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    for r in range(rank):
        if len(U[r]) != 1 or len(V[r]) != 1 or len(W[r]) != 1:
            return False
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # T should equal 1 if i,j,k all valid, else 0
    # Actually the tensor is: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait, the matrix multiplication tensor is: (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid i,j,k
    # Let's verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(rank, max_iters=50, time_limit=30.0):
    """Try to find a rank-`rank` decomposition via numerical optimization."""
    n, m, p = 3, 3, 6
    t0 = time.time()
    
    # Initialize near schoolbook (take first `rank` columns)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Use schoolbook initialization for first `rank` terms
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col >= rank:
                    break
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
            if col >= rank:
                break
        if col >= rank:
            break
    
    # Gradient descent on the loss: sum_{i,j,k} (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
    lr = 0.01
    best_loss = float('inf')
    
    for it in range(max_iters):
        if time.time() - t0 > time_limit:
            break
        
        # Compute residuals
        residual = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    residual[i, j, k] = s - 1.0
        
        loss = np.sum(residual ** 2)
        if loss < best_loss:
            best_loss = loss
            if loss < 1e-12:
                break
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r_ijk = residual[i, j, k]
                    for r in range(rank):
                        dU[i * m + j, r] += 2 * r_ijk * V[j * p + k, r] * W[k * n + i, r]
                        dV[j * p + k, r] += 2 * r_ijk * U[i * m + j, r] * W[k * n + i, r]
                        dW[k * n + i, r] += 2 * r_ijk * U[i * m + j, r] * V[j * p + k, r]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Decay learning rate
        lr *= 0.95
    
    # Check if we found a valid solution
    if best_loss < 1e-9:
        # Convert to list of lists
        U_list = [[float(U[i, r]) for r in range(rank)] for i in range(n * m)]
        V_list = [[float(V[i, r]) for r in range(rank)] for i in range(m * p)]
        W_list = [[float(W[i, r]) for r in range(rank)] for i in range(p * n)]
        if _verify(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    
    # Try to find a lower-rank decomposition
    # Known lower bounds: for 3x3x6, the rank is at least max(3*3, 3*6, 6*3) = 18
    # But the actual rank is likely 54 (schoolbook) or slightly less
    # Try ranks from 53 down to 50 with limited time
    for rank in [53, 52, 51, 50]:
        if time.time() > deadline - 5:
            break
        result = _try_rank(rank, max_iters=30, time_limit=25.0)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
