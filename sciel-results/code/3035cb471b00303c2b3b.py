import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Build valid and invalid index sets
    valid_set = set()
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                valid_set.add((alpha, beta, gamma))
                valid_indices.append((alpha, beta, gamma))
    
    all_indices = list(product(range(n * m), range(m * p), range(p * n)))
    invalid_indices = [idx for idx in all_indices if idx not in valid_set]
    
    def verify(U, V, W):
        r = U.shape[1]
        # Compute full tensor efficiently
        T = np.einsum('ac,bc,dc->abd', U, V, W)
        for (a, b, g) in valid_indices:
            if abs(T[a, b, g] - 1.0) > 1e-8:
                return False
        for (a, b, g) in invalid_indices:
            if abs(T[a, b, g]) > 1e-8:
                return False
        return True
    
    def snap_half_int(arr):
        return np.round(arr * 2) / 2
    
    def try_rank(r, max_restarts=50):
        """Try to find a rank-r decomposition"""
        U_shape = (n * m, r)
        V_shape = (m * p, r)
        W_shape = (p * n, r)
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Sparse random initialization with half-integers
            density = max(0.1, 0.5 - 0.02 * r)
            vals = np.array([-1.0, -0.5, 0.5, 1.0])
            
            U = np.zeros(U_shape)
            V = np.zeros(V_shape)
            W = np.zeros(W_shape)
            
            for idx in np.where(np.random.rand(*U_shape) < density):
                U[idx] = vals[np.random.randint(0, len(vals))]
            for idx in np.where(np.random.rand(*V_shape) < density):
                V[idx] = vals[np.random.randint(0, len(vals))]
            for idx in np.where(np.random.rand(*W_shape) < density):
                W[idx] = vals[np.random.randint(0, len(vals))]
            
            # Gradient descent with momentum
            lr = 0.01
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            best_loss = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for step in range(2000):
                if time.time() > deadline:
                    return None
                
                # Compute tensor
                T = np.einsum('ac,bc,dc->abd', U, V, W)
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                loss = 0.0
                
                for (a, b, g) in valid_indices:
                    err = T[a, b, g] - 1.0
                    loss += err * err
                    grad_U[a, :] += 2 * err * V[b, :] * W[g, :]
                    grad_V[b, :] += 2 * err * U[a, :] * W[g, :]
                    grad_W[g, :] += 2 * err * U[a, :] * V[b, :]
                
                for (a, b, g) in invalid_indices:
                    err = T[a, b, g]
                    loss += err * err
                    grad_U[a, :] += 2 * err * V[b, :] * W[g, :]
                    grad_V[b, :] += 2 * err * U[a, :] * W[g, :]
                    grad_W[g, :] += 2 * err * U[a, :] * V[b, :]
                
                # Update with momentum
                vU = momentum * vU - lr * grad_U
                vV = momentum * vV - lr * grad_V
                vW = momentum * vW - lr * grad_W
                U += vU
                V += vV
                W += vW
                
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if loss < 1e-10:
                    break
            
            # Try to snap to half-integers
            for scale in [1, 2, 4]:
                U_s = snap_half_int(best_U * scale) / scale
                V_s = snap_half_int(best_V * scale) / scale
                W_s = snap_half_int(best_W * scale) / scale
                if verify(U_s, V_s, W_s):
                    return (U_s, V_s, W_s)
            
            # Also try without snapping if loss is very low
            if best_loss < 1e-12:
                if verify(best_U, best_V, best_W):
                    return (best_U, best_V, best_W)
        
        return None
    
    # Try ranks from low to high
    for r in range(20, 37):
        if time.time() > deadline:
            break
        result = try_rank(r, max_restarts=30)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
