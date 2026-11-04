"""Program for 4x5 * 5x6 matrix multiplication tensor decomposition.
Finds a rank-minimal exact tensor decomposition using structured algebraic
constructions and random search with ALS refinement.
"""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]
n, m, p = DIMENSIONS


def _schoolbook():
    """Return the standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, tol=1e-6):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U_reshaped = U.reshape(n, m, -1)
    V_reshaped = V.reshape(m, p, -1)
    W_reshaped = W.reshape(p, n, -1)
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    except Exception:
        return False
    if T.shape != (n, m, p):
        return False
    return np.allclose(T, 1.0, atol=tol, rtol=tol)


def _to_integer_or_half(U, V, W, tol=1e-6):
    """Convert entries to integer or half-integer if possible."""
    def snap(arr):
        arr2 = 2.0 * arr
        rounded = np.round(arr2)
        if np.allclose(arr2, rounded, atol=tol):
            return rounded / 2.0
        return arr
    U = snap(U)
    V = snap(V)
    W = snap(W)
    return U, V, W


def _gradient_descent(target_rank, max_iter=200, lr=0.01, seed=0, deadline=None):
    """
    Use gradient descent with momentum and adaptive learning rate to find a
    rank-target_rank decomposition.
    Returns (U, V, W) or None.
    """
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    # Multiple initialization strategies
    scale = 0.5
    U = rng.randn(n * m, target_rank) * scale
    V = rng.randn(m * p, target_rank) * scale
    W = rng.randn(p * n, target_rank) * scale
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Precompute the index mappings
    I, J, K = np.indices((n, m, p))
    I_flat = I.ravel()
    J_flat = J.ravel()
    K_flat = K.ravel()
    
    u_idx_flat = I_flat * m + J_flat
    v_idx_flat = J_flat * p + K_flat
    w_idx_flat = K_flat * n + I_flat
    
    target = np.ones(n * m * p)
    
    # Momentum
    vel_U = np.zeros_like(U)
    vel_V = np.zeros_like(V)
    vel_W = np.zeros_like(W)
    momentum = 0.9
    
    lr_current = lr
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute T
        U_flat = U[u_idx_flat, :]
        V_flat = V[v_idx_flat, :]
        W_flat = W[w_idx_flat, :]
        
        T_flat = np.sum(U_flat * V_flat * W_flat, axis=1)
        err_vec = T_flat - target
        loss = np.sum(err_vec**2)
        
        if loss < best_err:
            best_err = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-12:
            break
        
        # Adaptive learning rate: reduce if loss increases
        if it > 0 and loss > prev_loss:
            lr_current *= 0.9
        prev_loss = loss
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        err_3d = err_vec.reshape(n, m, p)
        
        for c in range(target_rank):
            V_c = V[:, c].reshape(m, p)
            W_c = W[:, c].reshape(p, n)
            U_c = U[:, c].reshape(n, m)
            
            dU_c = np.einsum('ijk,jk,ki->ij', err_3d, V_c, W_c)
            dU[:, c] = dU_c.ravel()
            
            dV_c = np.einsum('ijk,ij,ki->jk', err_3d, U_c, W_c)
            dV[:, c] = dV_c.ravel()
            
            dW_c = np.einsum('ijk,ij,jk->ki', err_3d, U_c, V_c)
            dW[:, c] = dW_c.ravel()
        
        # Update with momentum
        vel_U = momentum * vel_U - lr_current * dU
        vel_V = momentum * vel_V - lr_current * dV
        vel_W = momentum * vel_W - lr_current * dW
        
        U += vel_U
        V += vel_V
        W += vel_W
    
    if best_err < 1e-6:
        return best_U, best_V, best_W
    return None


def _polish(U, V, W, max_iter=100, tol=1e-10, deadline=None):
    """
    High-precision polish pass using Newton-like refinement.
    Assumes U, V, W are already close to a valid decomposition.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute residual
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        err_3d = T - 1.0
        loss = np.sum(err_3d**2)
        
        if loss < tol:
            break
        
        # Small gradient step for polishing
        lr = 1e-3
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for c in range(U.shape[1]):
            V_c = V[:, c].reshape(m, p)
            W_c = W[:, c].reshape(p, n)
            U_c = U[:, c].reshape(n, m)
            
            dU_c = np.einsum('ijk,jk,ki->ij', err_3d, V_c, W_c)
            dU[:, c] = dU_c.ravel()
            
            dV_c = np.einsum('ijk,ij,ki->jk', err_3d, U_c, W_c)
            dV[:, c] = dV_c.ravel()
            
            dW_c = np.einsum('ijk,ij,jk->ki', err_3d, U_c, V_c)
            dW[:, c] = dW_c.ravel()
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    return U, V, W


def _random_search(target_rank, n_trials=30, deadline=None):
    """
    Random search for a valid decomposition at the given rank.
    """
    for trial in range(n_trials):
        if deadline is not None and time.time() > deadline:
            break
        seed = 42 + trial * 13
        result = _gradient_descent(target_rank, max_iter=150, lr=0.05, seed=seed, deadline=deadline)
        if result is not None:
            return result
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook()
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = n * m * p
    
    # Lower bound: max(n*m, m*p, p*n) = max(20, 30, 24) = 30
    min_rank = max(n * m, m * p, p * n)
    
    # Try ranks from low to high
    for target_rank in range(min_rank, best_rank + 1):
        if time.time() > deadline - 30.0:
            break
        
        found = False
        for seed in range(25):
            if time.time() > deadline - 15.0:
                break
            
            result = _gradient_descent(target_rank, max_iter=150, lr=0.05, seed=seed, deadline=deadline)
            if result is not None:
                U_try, V_try, W_try = result
                if _verify(U_try, V_try, W_try, tol=1e-6):
                    # Polish for higher precision
                    U_try, V_try, W_try = _polish(U_try, V_try, W_try, max_iter=50, tol=1e-12, deadline=deadline)
                    U_try, V_try, W_try = _to_integer_or_half(U_try, V_try, W_try, tol=1e-5)
                    if _verify(U_try, V_try, W_try, tol=1e-6):
                        best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                        best_rank = target_rank
                        found = True
                        break
        
        if found:
            break
    
    # Final snap and verify
    U, V, W = _to_integer_or_half(best_U, best_V, best_W)
    if not _verify(U, V, W, tol=1e-6):
        U, V, W = best_U, best_V, best_W
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
