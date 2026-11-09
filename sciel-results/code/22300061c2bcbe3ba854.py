"""Optimized tensor decomposition for 3x3 x 3x4 matrix multiplication."""
DIMENSIONS = [3, 3, 4]

import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    tensor = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Start with schoolbook rank 36
    best_rank = n * m * p
    best_U = np.zeros((n*m, best_rank))
    best_V = np.zeros((m*p, best_rank))
    best_W = np.zeros((p*n, best_rank))
    for c in range(best_rank):
        i = c // (m*p)
        j = (c // p) % m
        k = c % p
        best_U[i*m + j, c] = 1
        best_V[j*p + k, c] = 1
        best_W[k*n + i, c] = 1
    
    def round_to_half(x):
        return np.round(2*x) / 2.0
    
    def verify_decomposition(U, V, W):
        """Check if U, V, W exactly decompose the tensor (with tolerance)."""
        rank = U.shape[1]
        for i in range(n*m):
            for a in range(m*p):
                for b in range(p*n):
                    val = sum(U[i, c] * V[a, c] * W[b, c] for c in range(rank))
                    if abs(val - tensor[i, a, b]) > 1e-6:
                        return False
        return True
    
    def try_find_decomposition(rank, max_iter=200, timeout=60):
        """Try to find a rank decomposition with given rank."""
        t0 = time.time()
        np.random.seed(rank * 1000 + 42)
        
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            if time.time() - t0 > timeout:
                break
            
            # Initialize with random values
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # Use the schoolbook structure as warm start for some attempts
            if attempt == 0 and rank <= best_rank:
                # Try to find a subset of schoolbook terms
                pass
            
            def loss_and_grad(params):
                if time.time() > deadline - 5:
                    return 0, None
                U_f = params[0:n*m*rank].reshape(n*m, rank)
                V_f = params[n*m*rank:2*n*m*rank].reshape(m*p, rank)
                W_f = params[2*n*m*rank:3*n*m*rank].reshape(p*n, rank)
                
                # Compute tensor contraction
                loss = 0.0
                for i in range(n*m):
                    for a in range(m*p):
                        for b in range(p*n):
                            val = 0.0
                            for c in range(rank):
                                val += U_f[i, c] * V_f[a, c] * W_f[b, c]
                            diff = val - tensor[i, a, b]
                            loss += diff * diff
                return loss, None
            
            # Use Nelder-Mead for small problems
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            result = minimize(loss_and_grad, params0, method='Nelder-Mead',
                             options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-8})
            
            if result.fun < 1e-6:
                params = result.x
                U_f = params[0:n*m*rank].reshape(n*m, rank)
                V_f = params[n*m*rank:2*n*m*rank].reshape(m*p, rank)
                W_f = params[2*n*m*rank:3*n*m*rank].reshape(p*n, rank)
                
                # Round to nearest half-integer
                U_r = round_to_half(U_f)
                V_r = round_to_half(V_f)
                W_r = round_to_half(W_f)
                
                if verify_decomposition(U_r, V_r, W_r):
                    return U_r, V_r, W_r
                
                # Try without rounding - maybe values are already clean
                if verify_decomposition(U_f, V_f, W_f):
                    return U_f, V_f, W_f
        
        return None
    
    # Try to find lower rank decompositions
    for rank in range(28, best_rank):  # Try from 28 up to 35
        if time.time() > deadline - 15:
            break
        result = try_find_decomposition(rank)
        if result is not None:
            U, V, W = result
            if verify_decomposition(U, V, W):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break
    
    # Convert to lists
    U_list = [[float(best_U[i, c]) for c in range(best_rank)] for i in range(n*m)]
    V_list = [[float(best_V[i, c]) for c in range(best_rank)] for i in range(m*p)]
    W_list = [[float(best_W[i, c]) for c in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
