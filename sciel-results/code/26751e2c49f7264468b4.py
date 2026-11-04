"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
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

def _try_structured_rank(n, m, p, rank, seed=42):
    """Try structured decomposition with integer/half-integer entries."""
    np.random.seed(seed)
    random.seed(seed)
    
    # Use small integer values for better chance of exact solutions
    vals = [0, 1, -1, 2, -2]
    
    U = [[random.choice(vals) for _ in range(rank)] for _ in range(n * m)]
    V = [[random.choice(vals) for _ in range(rank)] for _ in range(m * p)]
    W = [[random.choice(vals) for _ in range(rank)] for _ in range(p * n)]
    
    return U, V, W

def _try_kronecker_like(n, m, p, rank):
    """Try a Kronecker-product-inspired structured decomposition."""
    # For matrix multiplication, we can try to exploit structure
    # The tensor T_{i,j,k} = delta_{i,j,k} for matmul has specific structure
    
    # Try a factorization based on splitting dimensions
    # This is heuristic - try to find patterns
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Attempt: use rank = min(n*m*p, some structured value)
    # For now, return None if we can't construct one
    return None

def _gradient_descent(n, m, p, rank, max_iters=3000, seed=42, lr_init=0.1):
    """Gradient descent to find a rank-r decomposition."""
    np.random.seed(seed)
    
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = lr_init
    for it in range(max_iters):
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        err_sum = 0.0
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = U[i * m + j]
                    vj = V[j * p + k]
                    wk = W[k * n + i]
                    s = np.dot(ui, vj * wk)
                    grad = s - 1.0
                    err_sum += grad * grad
                    dU[i * m + j] += grad * vj * wk
                    dV[j * p + k] += grad * ui * wk
                    dW[k * n + i] += grad * ui * vj
        
        err = np.sqrt(err_sum / (n * m * p))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-10:
            break
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        if it % 200 == 199:
            lr *= 0.8
    
    if best_err < 1e-9:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return U_list, V_list, W_list
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 90
    
    # Known lower bound for matrix multiplication rank is roughly (n*m*p)^(2/3)
    # For 3x5x6: (90)^(2/3) ≈ 20.1, so try from ~25 upward
    # But also try some structured ranks
    
    # First, try structured random with small integers for various ranks
    # This is fast and might find exact solutions
    for rank in [30, 35, 40, 45, 50, 55, 60]:
        if time.time() > deadline - 20:
            break
        
        found = None
        for seed in range(20):
            if time.time() > deadline - 20:
                break
            U, V, W = _try_structured_rank(n, m, p, rank, seed=seed)
            if _verify(U, V, W, n, m, p):
                found = (U, V, W)
                break
        
        if found is not None:
            U, V, W = found
            best_rank = rank
            best = {"U": U, "V": V, "W": W}
            # Don't break - try to find even lower rank
            continue
    
    # If we found something better than schoolbook, try to improve further
    if best_rank < n * m * p:
        for rank in range(max(20, best_rank - 5), best_rank):
            if time.time() > deadline - 10:
                break
            
            found = None
            for seed in range(10):
                if time.time() > deadline - 10:
                    break
                U, V, W = _try_structured_rank(n, m, p, rank, seed=seed)
                if _verify(U, V, W, n, m, p):
                    found = (U, V, W)
                    break
            
            if found is not None:
                U, V, W = found
                best_rank = rank
                best = {"U": U, "V": V, "W": W}
    
    # Try gradient descent for ranks that structured search didn't find
    if time.time() < deadline - 15:
        for rank in range(max(25, best_rank - 10), best_rank + 1):
            if time.time() > deadline - 5:
                break
            
            found = None
            for seed in range(3):
                if time.time() > deadline - 5:
                    break
                result = _gradient_descent(n, m, p, rank, max_iters=1500, seed=seed)
                if result is not None:
                    U, V, W = result
                    if _verify(U, V, W, n, m, p):
                        found = (U, V, W)
                        break
            
            if found is not None:
                U, V, W = found
                if rank < best_rank:
                    best_rank = rank
                    best = {"U": U, "V": V, "W": W}
    
    return best
# EVOLVE-BLOCK-END
