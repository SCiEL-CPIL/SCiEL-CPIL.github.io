"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _try_optimize_rank(n, m, p, target_rank, max_iters=500, seed=42):
    """Try to find a rank-target_rank decomposition using optimization."""
    np.random.seed(seed)
    nm, mp, pn = n*m, m*p, p*n
    
    # Initialize with random small values
    U = np.random.randn(nm, target_rank) * 0.1
    V = np.random.randn(mp, target_rank) * 0.1
    W = np.random.randn(pn, target_rank) * 0.1
    
    # Build the tensor equation: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is a system of n*m*p equations with nm*rank + mp*rank + pn*rank unknowns
    
    # Use gradient descent on the loss: sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.01
    for it in range(max_iters):
        # Compute loss and gradients
        loss = 0.0
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i*m + j
                    idx_v = j*p + k
                    idx_w = k*n + i
                    
                    val = np.dot(U[idx_u], V[idx_v] * W[idx_w])
                    err = val - 1.0
                    loss += err * err
                    
                    # Gradient: d/dU = 2*err*V*W, d/dV = 2*err*U*W, d/dW = 2*err*U*V
                    gU[idx_u] += 2 * err * V[idx_v] * W[idx_w]
                    gV[idx_v] += 2 * err * U[idx_u] * W[idx_w]
                    gW[idx_w] += 2 * err * U[idx_u] * V[idx_v]
        
        # Update
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if loss < 1e-8:
            break
        
        # Decay learning rate
        if it % 100 == 99:
            lr *= 0.5
    
    # Check if we found a valid solution
    if best_loss < 1e-6:
        # Round to nearest half-integer
        U_r = np.round(best_U * 2) / 2
        V_r = np.round(best_V * 2) / 2
        W_r = np.round(best_W * 2) / 2
        
        # Verify
        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, tol=1e-6):
            return U_r.tolist(), V_r.tolist(), W_r.tolist(), best_loss
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 150s limit, leave 5s margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find lower rank decompositions
    for target_rank in range(best_rank - 1, 0, -1):
        if time.time() > deadline - 30:
            break
        
        result = _try_optimize_rank(n, m, p, target_rank, max_iters=200, seed=target_rank)
        if result is not None:
            U, V, W, loss = result
            best_rank = target_rank
            best_result = {"U": U, "V": V, "W": W}
            break
    
    return best_result

# EVOLVE-BLOCK-END
