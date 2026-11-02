import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    """
    Constructs the matrix multiplication tensor decomposition for 2x3 by 3x5.
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    
    # Precompute target tensor
    nm, mp, pn = n*m, m*p, p*n
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    def try_rank_grad(r, max_iters=2000, lr=0.003, seed=42):
        """Try to find a rank-r solution using gradient descent."""
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        U = np.random.randn(nm, r) * 0.1
        V = np.random.randn(mp, r) * 0.1
        W = np.random.randn(pn, r) * 0.1
        
        best_res = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute R = T_reconstructed - target
            R = np.zeros((nm, mp, pn))
            for r_idx in range(r):
                R += np.einsum('ar,br,cr->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
            R -= target
            
            res = float(np.sum(R ** 2))
            if res < best_res:
                best_res = res
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_res < 1e-10:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for r_idx in range(r):
                dU[:, r_idx] = np.einsum('abc,br,cr->a', R, V[:, r_idx], W[:, r_idx])
                dV[:, r_idx] = np.einsum('abc,ar,cr->b', R, U[:, r_idx], W[:, r_idx])
                dW[:, r_idx] = np.einsum('abc,ar,br->c', R, U[:, r_idx], V[:, r_idx])
            
            # Adaptive learning rate
            grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
            if grad_norm < 1e-12:
                break
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        if best_res < 1e-6:
            U_list = [[float(x) for x in row] for row in best_U]
            V_list = [[float(x) for x in row] for row in best_V]
            W_list = [[float(x) for x in row] for row in best_W]
            return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 25 to 29
    for r in range(25, 30):
        if time.time() > deadline:
            break
        
        for seed in range(5):
            if time.time() > deadline:
                break
            result = try_rank_grad(r, max_iters=1500, lr=0.003, seed=seed)
            if result is not None:
                return result
    
    return best_result
