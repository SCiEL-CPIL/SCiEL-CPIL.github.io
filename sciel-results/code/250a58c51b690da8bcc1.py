"""Matrix multiplication 3,3,6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]
DEADLINE = 135.0  # seconds, safety margin before 150s kill

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W, "rank": rank}


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # For each (i,j,k), check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = np.sum(U_np[i * m + j, :] * V_np[j * p + k, :] * W_np[k * n + i, :])
                if abs(actual - 1.0) > tol:
                    return False
    return True


def _try_structured_decomposition(n, m, p, target_rank, seed=42):
    """
    Try to find a decomposition using structured initialization.
    For (3,3,6), exploit the block structure: k=0,1,2 is block 0, k=3,4,5 is block 1.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    np.random.seed(seed)
    
    U = np.zeros((nm, target_rank))
    V = np.zeros((mp, target_rank))
    W = np.zeros((pn, target_rank))
    
    # Split target_rank into two halves for the two blocks
    half_rank = target_rank // 2
    rem = target_rank % 2
    
    for block in range(2):
        start = block * half_rank + min(block, rem)
        end = start + half_rank + (1 if block < rem else 0)
        block_size = end - start
        
        if block_size <= 0:
            continue
            
        # Random initialization for this block
        U_block = np.random.randn(nm, block_size)
        V_block = np.zeros((mp, block_size))
        W_block = np.zeros((pn, block_size))
        
        # Only activate relevant indices
        k_indices = [3 * block + b for b in range(3)]
        
        # Fill V and W for active k indices
        for k in k_indices:
            for j in range(m):
                V_block[j * p + k, :] = np.random.randn(block_size)
            for i in range(n):
                W_block[k * n + i, :] = np.random.randn(block_size)
        
        U[start:end] = U_block
        V[:, start:end] = V_block
        W[:, start:end] = W_block
    
    # Gradient descent to minimize loss
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(200):
        if time.time() > DEADLINE - 10:
            break
        
        # Compute loss and gradients
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Expected: 1.0
                    actual = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    err = actual - 1.0
                    loss += err ** 2
                    
                    # Gradients
                    grad_U[i * m + j, :] += 2 * err * V[j * p + k, :] * W[k * n + i, :]
                    grad_V[j * p + k, :] += 2 * err * U[i * m + j, :] * W[k * n + i, :]
                    grad_W[k * n + i, :] += 2 * err * U[i * m + j, :] * V[j * p + k, :]
        
        loss /= (n * m * p)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-12:
            break
        
        # Adaptive learning rate
        lr = 0.01 / (1.0 + best_loss)
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
    
    if best_loss < 1e-8:
        # Try rounding to half-integers
        U_rounded = np.round(best_U * 2.0) / 2.0
        V_rounded = np.round(best_V * 2.0) / 2.0
        W_rounded = np.round(best_W * 2.0) / 2.0
        
        if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p, tol=1e-6):
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist(), "rank": target_rank}
        
        # If rounding doesn't work, return unrounded if it verifies
        if _verify(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p, tol=1e-6):
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist(), "rank": target_rank}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Lower bound: max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower bound upward
    for target_rank in range(lower_bound, min(schoolbook_rank + 1, lower_bound + 40)):
        if time.time() > DEADLINE - 15:
            break
        
        # Try multiple seeds
        for seed in range(10):
            if time.time() > DEADLINE - 15:
                break
            result = _try_structured_decomposition(n, m, p, target_rank, seed=seed)
            if result is not None:
                return {"U": result["U"], "V": result["V"], "W": result["W"]}
    
    # Fall back to schoolbook
    sb = _schoolbook(n, m, p)
    return {"U": sb["U"], "V": sb["V"], "W": sb["W"]}

# EVOLVE-BLOCK-END
