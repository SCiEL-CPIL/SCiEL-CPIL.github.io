"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _loss_and_grad(U, V, W, n, m, p):
    """Compute loss and gradients for the tensor decomposition problem.
    
    Loss = sum_{i,j,k} (sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] - 1)^2
    """
    loss = 0.0
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute inner product
                inner = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                err = inner - 1.0
                loss += err * err
                
                # Gradients
                dU[i*m+j] += 2 * err * V[j*p+k] * W[k*n+i]
                dV[j*p+k] += 2 * err * U[i*m+j] * W[k*n+i]
                dW[k*n+i] += 2 * err * U[i*m+j] * V[j*p+k]
    
    return loss, dU, dV, dW

def _optimize(U, V, W, n, m, p, max_iters=5000, lr_init=0.01, time_limit=30):
    """Optimize using Adam with adaptive learning rate."""
    start_time = time.time()
    rank = U.shape[1]
    
    # Adam optimizer state
    mU = np.zeros_like(U)
    vU = np.zeros_like(U)
    mV = np.zeros_like(V)
    vV = np.zeros_like(V)
    mW = np.zeros_like(W)
    vW = np.zeros_like(W)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    t = 0
    
    best_loss = float('inf')
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        loss, dU, dV, dW = _loss_and_grad(U, V, W, n, m, p)
        
        if loss < best_loss:
            best_loss = loss
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if best_loss < 1e-10:
            break
        
        t += 1
        lr = lr_init / (1 + 0.001 * t)
        
        # Update U
        mU = b1 * mU + (1 - b1) * dU
        vU = b2 * vU + (1 - b2) * dU**2
        mU_hat = mU / (1 - b1**t)
        vU_hat = vU / (1 - b2**t)
        U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
        
        # Update V
        mV = b1 * mV + (1 - b1) * dV
        vV = b2 * vV + (1 - b2) * dV**2
        mV_hat = mV / (1 - b1**t)
        vV_hat = vV / (1 - b2**t)
        V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
        
        # Update W
        mW = b1 * mW + (1 - b1) * dW
        vW = b2 * vW + (1 - b2) * dW**2
        mW_hat = mW / (1 - b1**t)
        vW_hat = vW / (1 - b2**t)
        W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
    
    return best_U, best_V, best_W, best_loss

def _try_rank(n, m, p, target_rank, time_limit=30):
    """Try to find a decomposition of the given rank using random initialization and optimization."""
    start_time = time.time()
    
    # Try multiple random restarts
    for restart in range(10):
        if time.time() - start_time > time_limit:
            break
        
        np.random.seed(42 + restart)
        
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        U_opt, V_opt, W_opt, loss = _optimize(U, V, W, n, m, p, max_iters=3000, lr_init=0.01, time_limit=time_limit - (time.time() - start_time))
        
        if loss < 1e-8:
            # Verify and return
            if _verify(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p):
                return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    for target_rank in range(20, best_rank):
        if time.time() > deadline - 15:
            break
        
        result = _try_rank(n, m, p, target_rank, time_limit=25)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = target_rank
                break
    
    return best
# EVOLVE-BLOCK-END
