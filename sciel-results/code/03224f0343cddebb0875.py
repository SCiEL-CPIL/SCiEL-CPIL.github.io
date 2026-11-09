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
                
                val = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                diff = val - 1.0
                loss += diff * diff
                
                factor = 2.0 * diff
                grad_U[u_idx] += factor * V[v_idx] * W[w_idx]
                grad_V[v_idx] += factor * U[u_idx] * W[w_idx]
                grad_W[w_idx] += factor * U[u_idx] * V[v_idx]
    
    return loss, grad_U, grad_V, grad_W

def _polish(U, V, W, n, m, p, max_iter=200, tol=1e-14):
    """Final high-precision polish pass with adaptive learning rate."""
    lr = 0.01
    best_loss = np.inf
    
    for it in range(max_iter):
        loss, grad_U, grad_V, grad_W = _compute_loss_and_grads(U, V, W, n, m, p)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < tol:
            break
        
        # Adaptive learning rate based on gradient magnitude
        grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
        if grad_norm > 0:
            lr = 0.01 * (1.0 + 0.1 * it)  # Slightly increase lr
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
    
    return best_U, best_V, best_W, best_loss

def _try_rank(n, m, p, rank, max_iter=500, seed=42):
    """Try to find a rank-r decomposition using randomized local search."""
    rng = np.random.RandomState(seed)
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    best_loss = np.inf
    best_U, best_V, best_W = None, None, None
    
    # Try multiple initialization strategies
    for init_seed in range(3):
        init_rng = np.random.RandomState(seed + init_seed * 1000)
        
        # Different initialization scales
        scale = [0.1, 0.5, 1.0][init_seed % 3]
        U = init_rng.randn(num_u, rank) * scale
        V = init_rng.randn(num_v, rank) * scale
        W = init_rng.randn(num_w, rank) * scale
        
        lr = 0.005
        for it in range(max_iter):
            loss, grad_U, grad_V, grad_W = _compute_loss_and_grads(U, V, W, n, m, p)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-14:
                break
            
            # Adaptive learning rate
            grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
            if grad_norm > 1e-10:
                lr = 0.005 * (1.0 + 0.05 * (it % 50))
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
        
        if best_loss < 1e-14:
            break
    
    if best_U is not None and best_loss < 1e-14:
        # Final polish
        best_U, best_V, best_W, final_loss = _polish(best_U, best_V, best_W, n, m, p, max_iter=100)
        
        if final_loss < 1e-14:
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist(), "rank": rank}
    
    return None

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = DIMENSIONS
    best_decomp = _schoolbook()
    best_rank = best_decomp["rank"]
    
    for rank in range(40, best_rank):
        if time.time() > deadline:
            break
        
        # More seeds for lower ranks where search is harder
        num_seeds = 5 if rank < 50 else 3
        for seed in range(num_seeds):
            if time.time() > deadline:
                break
            
            result = _try_rank(n, m, p, rank, max_iter=300, seed=seed + 100)
            if result is not None:
                if _verify(result, n, m, p):
                    best_decomp = result
                    best_rank = rank
                    break
        if best_rank == rank:
            break
    
    return {"U": best_decomp["U"], "V": best_decomp["V"], "W": best_decomp["W"]}
# EVOLVE-BLOCK-END
