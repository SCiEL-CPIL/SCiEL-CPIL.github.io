"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

import numpy as np
import time
import random

def _build_target_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _validate_full(U, V, W, n, m, p, tol=1e-6):
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    T = _build_target_tensor(n, m, p)
    for c in range(p * n):
        M_c = (U * W[c, :]) @ V.T
        if np.max(np.abs(M_c - T[:, :, c])) > tol:
            return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _fast_loss_and_grad(U, V, W, T):
    """Compute loss and gradients efficiently using matrix operations."""
    n_rows, rank = U.shape
    m_cols, _ = V.shape
    p_n, _ = W.shape
    
    loss = 0.0
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    for c in range(p_n):
        M_c = (U * W[c, :]) @ V.T
        err = M_c - T[:, :, c]
        loss += np.sum(err * err)
        
        eV = err @ V
        dU += 2.0 * (eV * W[c, :])
        
        eU = err.T @ U
        dV += 2.0 * (eU * W[c, :])
        
        dW[c, :] = 2.0 * np.sum(U * eV, axis=0)
    
    return loss, dU, dV, dW

def _optimize_rank(U, V, W, T, max_time, tol=1e-10):
    """Optimize a rank-r decomposition using adaptive gradient descent."""
    start = time.time()
    U = U.astype(np.float64).copy()
    V = V.astype(np.float64).copy()
    W = W.astype(np.float64).copy()
    rank = U.shape[1]
    
    lr = 0.05
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(2000):
        if time.time() - start > max_time - 1:
            break
        
        loss, dU, dV, dW = _fast_loss_and_grad(U, V, W, T)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Adaptive learning rate
        grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
        if grad_norm < 1e-12:
            break
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Decay learning rate
        if it % 100 == 99:
            lr *= 0.8
        
        if loss < tol:
            break
    
    return best_U, best_V, best_W, best_loss

def _try_reduce_rank(U, V, W, T, n, m, p, max_time):
    """Try to reduce rank by finding near-zero columns."""
    start = time.time()
    rank = U.shape[1]
    
    for r in range(rank - 1, -1, -1):
        if time.time() - start > max_time - 5:
            break
        if np.max(np.abs(U[:, r])) < 1e-8 and \
           np.max(np.abs(V[:, r])) < 1e-8 and \
           np.max(np.abs(W[:, r])) < 1e-8:
            U = np.delete(U, r, axis=1)
            V = np.delete(V, r, axis=1)
            W = np.delete(W, r, axis=1)
            return U, V, W, True
    
    return U, V, W, False

def _construct_335():
    n, m, p = 3, 3, 5
    start_time = time.time()
    deadline = start_time + 135
    
    T = _build_target_tensor(n, m, p)
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = 45
    best_result = (U.copy(), V.copy(), W.copy())
    
    # Try various target ranks with multiple restarts
    target_ranks = [35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
    
    rng = np.random.RandomState(12345)
    
    for target_rank in target_ranks:
        if time.time() > deadline - 10:
            break
        
        if target_rank >= best_rank:
            continue
        
        max_restarts = 3 if target_rank <= 37 else 2
        time_per_restart = (deadline - time.time() - 10) / (max_restarts * len([r for r in target_ranks if r < target_rank + 1]))
        time_per_restart = min(time_per_restart, 15)
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Initialize: start from schoolbook and zero out random columns, then optimize
            U_try = U[:, :target_rank].copy()
            V_try = V[:, :target_rank].copy()
            W_try = W[:, :target_rank].copy()
            
            # Add small random perturbation to avoid local minima
            U_try += rng.randn(9, target_rank) * 0.01
            V_try += rng.randn(15, target_rank) * 0.01
            W_try += rng.randn(15, target_rank) * 0.01
            
            remaining = deadline - time.time() - 8
            if remaining < 2:
                break
            
            U_opt, V_opt, W_opt, loss = _optimize_rank(U_try, V_try, W_try, T, remaining)
            
            if loss < 1e-6:
                if _validate_full(U_opt, V_opt, W_opt, n, m, p, tol=1e-5):
                    # Try to reduce rank further
                    U_red, V_red, W_red, reduced = _try_reduce_rank(U_opt, V_opt, W_opt, T, n, m, p, 5)
                    if reduced:
                        if _validate_full(U_red, V_red, W_red, n, m, p, tol=1e-5):
                            best_result = (U_red, V_red, W_red)
                            best_rank = U_red.shape[1]
                    else:
                        best_result = (U_opt, V_opt, W_opt)
                        best_rank = target_rank
                    break
        
        if best_rank <= target_rank:
            break
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    result = _construct_335()
    U, V, W = result
    
    if time.time() > deadline - 3:
        U, V, W = _schoolbook(n, m, p)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
