"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W, "rank": rank}

def _verify(decomp, n, m, p):
    """Verify that the tensor decomposition exactly multiplies the matrices."""
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    r = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for idx in range(r):
                    val += U[i*m + j][idx] * V[j*p + k][idx] * W[k*n + i][idx]
                if not (abs(val - 1.0) < 1e-9):
                    return False
    return True

def _compute_loss_and_grads(U, V, W, n, m, p):
    """Compute loss and gradients for the tensor decomposition."""
    num_u = n * m
    num_v = m * p
    num_w = p * n
    rank = U.shape[1]
    
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    loss = 0.0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                diff = val - 1.0
                loss += diff * diff
                
                factor = 2.0 * diff
                grad_U[u_idx] += factor * (V[v_idx] * W[w_idx])
                grad_V[v_idx] += factor * (U[u_idx] * W[w_idx])
                grad_W[w_idx] += factor * (U[u_idx] * V[v_idx])
    
    return loss, grad_U, grad_V, grad_W

def _try_rank(n, m, p, rank, max_iter=500, seed=42, time_limit=10.0):
    """Try to find a rank-r decomposition using randomized local search."""
    start_time = time.time()
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    # Try multiple seeds
    best_loss = np.inf
    best_U, best_V, best_W = None, None, None
    
    for s in range(5):
        if time.time() - start_time > time_limit:
            break
        rng = np.random.RandomState(seed + s)
        
        # Initialize with small random values
        U = rng.randn(num_u, rank) * 0.1
        V = rng.randn(num_v, rank) * 0.1
        W = rng.randn(num_w, rank) * 0.1
        
        lr = 0.005
        for it in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            
            loss, grad_U, grad_V, grad_W = _compute_loss_and_grads(U, V, W, n, m, p)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-12:
                return {"U": best_U, "V": best_V, "W": best_W, "rank": rank}
            
            # Adaptive learning rate
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Occasionally adjust learning rate
            if it % 50 == 0:
                if loss < best_loss * 0.9:
                    lr *= 1.2
                else:
                    lr *= 0.8
                    U += 0.5 * lr * grad_U  # Step back a bit
                    V += 0.5 * lr * grad_V
                    W += 0.5 * lr * grad_W
    
    if best_loss < 1e-10:
        return {"U": best_U, "V": best_V, "W": best_W, "rank": rank}
    
    return None

def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds safety margin
    
    n, m, p = DIMENSIONS
    best_decomp = _schoolbook()
    best_rank = best_decomp["rank"]
    
    # Try ranks from 40 up to schoolbook rank
    for rank in range(40, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple seeds for this rank
        for seed in range(3):
            if time.time() > deadline:
                break
            
            result = _try_rank(n, m, p, rank, max_iter=300, seed=seed + 100, time_limit=15.0)
            if result is not None:
                if _verify(result, n, m, p):
                    best_decomp = result
                    best_rank = rank
                    break
        if best_rank == rank:
            break
    
    return {"U": best_decomp["U"], "V": best_decomp["V"], "W": best_decomp["W"]}
# EVOLVE-BLOCK-END
