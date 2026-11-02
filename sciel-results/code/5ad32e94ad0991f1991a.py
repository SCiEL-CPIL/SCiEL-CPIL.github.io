import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    
    def compute_residual(U, V, W):
        """Compute the exact residual for a given decomposition."""
        res = 0.0
        for i in range(nm):
            a, b = divmod(i, m)
            for j in range(mp):
                c, d = divmod(j, p)
                k = d * n + a
                val = np.dot(U[i, :], V[j, :] * W[k, :])
                if b == c:
                    res += (val - 1.0) ** 2
                else:
                    res += val ** 2
        return res
    
    def verify(U, V, W, tol=1e-8):
        """Verify if decomposition is exact."""
        for i in range(nm):
            a, b = divmod(i, m)
            for j in range(mp):
                c, d = divmod(j, p)
                k = d * n + a
                val = np.dot(U[i, :], V[j, :] * W[k, :])
                if b == c:
                    if abs(val - 1.0) > tol:
                        return False
                else:
                    if abs(val) > tol:
                        return False
        return True
    
    # Try to find lower rank using structured initialization
    # Key idea: use the fact that the tensor is sparse and has structure
    
    # Strategy 1: Try to merge schoolbook columns using linear dependencies
    # The schoolbook has 84 terms. Can we find a smaller set that spans the same tensor?
    
    # Strategy 2: Use random initialization with optimization for lower ranks
    
    def try_optimize(R, n_restarts=10, max_iter=200):
        """Try to find rank R decomposition using optimization."""
        best_res = np.inf
        best_params = None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize with small random values
            U = np.random.randn(nm, R) * 0.5
            V = np.random.randn(mp, R) * 0.5
            W = np.random.randn(pn, R) * 0.5
            
            lr = 0.05
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                    
                # Compute residual and gradients using all entries
                # For efficiency, compute in blocks
                err_U = np.zeros((nm, R))
                err_V = np.zeros((mp, R))
                err_W = np.zeros((pn, R))
                
                for i in range(nm):
                    a, b = divmod(i, m)
                    for j in range(mp):
                        c, d = divmod(j, p)
                        k = d * n + a
                        uv = U[i, :] * V[j, :]
                        val = np.dot(uv, W[k, :])
                        target = 1.0 if b == c else 0.0
                        err = val - target
                        
                        # Gradients
                        err_U[i, :] += err * (V[j, :] * W[k, :])
                        err_V[j, :] += err * (U[i, :] * W[k, :])
                        err_W[k, :] += err * (U[i, :] * V[j, :])
                
                # Update
                U -= lr * err_U
                V -= lr * err_V
                W -= lr * err_W
                
                res = compute_residual(U, V, W)
                if res < best_res:
                    best_res = res
                    best_params = (U.copy(), V.copy(), W.copy())
                
                if res < 1e-12:
                    break
                
                # Adaptive learning rate
                if it > 0 and res > best_res * 1.1:
                    lr *= 0.5
                
        return best_res, best_params
    
    # Try ranks from 83 down to a reasonable minimum
    # Lower bound: max(n*m, m*p, p*n) = 28, but better bounds exist
    # For 3x4x7, the best known upper bound is 84 (schoolbook)
    # Let's try to find if we can do better
    
    min_rank = 60
    for R in range(rank_sb - 1, min_rank - 1, -1):
        if time.time() > deadline - 10:
            break
        
        res, params = try_optimize(R, n_restarts=5, max_iter=100)
        
        if res < 1e-10 and params is not None:
            U, V, W = params
            if verify(U, V, W, tol=1e-8):
                best_rank = R
                best_U, best_V, best_W = U, V, W
                break
    
    # If we found a better rank, return it; otherwise return schoolbook
    if best_rank < rank_sb:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
    else:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
