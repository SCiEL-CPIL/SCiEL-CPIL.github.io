import time
import random
import numpy as np
from itertools import product
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct."""
    T = _build_tensor(n, m, p)
    r = U.shape[1]
    T_recon = np.zeros_like(T)
    for c in range(r):
        T_recon += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
    return np.max(np.abs(T_recon - T)) < tol

def _try_rank(n, m, p, rank, seed=0, max_iters=2000, lr=0.01):
    """Try to find a rank-`rank` decomposition using gradient descent."""
    rng = np.random.RandomState(seed)
    T = _build_tensor(n, m, p)
    
    # Initialize with small random values
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Compute error
        T_recon = np.zeros_like(T)
        for c in range(rank):
            T_recon += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        err = np.sum((T_recon - T) ** 2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-12:
            break
        
        # Compute gradients
        E = T_recon - T
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for c in range(rank):
            dU[:, c] = 2.0 * np.einsum('abc,b,d->a', E, V[:, c], W[:, c])
            dV[:, c] = 2.0 * np.einsum('abc,a,d->b', E, U[:, c], W[:, c])
            dW[:, c] = 2.0 * np.einsum('abc,a,b->c', E, U[:, c], V[:, c])
        
        # Adaptive learning rate?
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Renormalize
        for c in range(rank):
            nU = np.linalg.norm(U[:, c])
            nV = np.linalg.norm(V[:, c])
            nW = np.linalg.norm(W[:, c])
            if nU > 1e6: U[:, c] /= nU
            if nV > 1e6: V[:, c] /= nV
            if nW > 1e6: W[:, c] /= nW
    
    return best_U, best_V, best_W, best_err

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Start with schoolbook
    rank_schoolbook = n * m * p  # 54
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    best_result = (U_sb, V_sb, W_sb, rank_schoolbook)
    
    start_time = time.time()
    deadline = start_time + 120
    
    # Try ranks from 40 to 53
    for rank in range(40, 54):
        if time.time() > deadline - 10:
            break
        
        found = False
        for seed in range(50):
            if time.time() > deadline - 10:
                break
            U, V, W, err = _try_rank(n, m, p, rank, seed=seed, max_iters=1000, lr=0.01)
            if err < 1e-10:
                if _verify(U, V, W, n, m, p, tol=1e-9):
                    best_result = (U, V, W, rank)
                    found = True
                    break
        
        if found:
            break
    
    U, V, W, rank = best_result
    
    # Snap to integers/half-integers
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    # Final verification
    if not _verify(U, V, W, n, m, p, tol=1e-9):
        # If snapping broke it, return the original float solution?
        # The task requires integer/half-integer.
        # If we can't satisfy it, we might return the schoolbook which is integer.
        U, V, W = U_sb, V_sb, W_sb
        rank = rank_schoolbook
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
