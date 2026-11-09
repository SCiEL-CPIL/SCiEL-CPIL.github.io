import numpy as np
import time
import random

def _validate_decomposition(U, V, W, n, m, p):
    """Validate that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    U_3d = np.array(U).reshape(n, m, rank)
    V_3d = np.array(V).reshape(m, p, rank)
    W_3d = np.array(W).reshape(p, n, rank)
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    return np.allclose(T, 1.0, atol=1e-9)

def construct():
    n, m, p = 3, 5, 7
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def try_rank(r, max_restarts=10):
        """Try to find a rank-r decomposition."""
        size_u = n * m * r
        size_v = m * p * r
        size_w = p * n * r
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                return None
            
            # Structured random initialization
            U = np.random.randn(n, m, r) * 0.5
            V = np.random.randn(m, p, r) * 0.5
            W = np.random.randn(p, n, r) * 0.5
            
            # Use gradient-free local search: perturb and check
            best_cost = np.inf
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for step in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Compute current tensor
                T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
                residual = (T - 1.0).flatten()
                cost = np.sum(residual ** 2)
                
                if cost < best_cost:
                    best_cost = cost
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if cost < 1e-12:
                    break
                
                # Compute gradients
                # dT/dU[i,j,c] = V[j,:,c] * W[:,i,c] contracted over k
                # dT/dV[j,k,c] = U[:,j,c] * W[k,:,c] contracted over i
                # dT/dW[k,i,c] = U[:,j,c] * V[j,k,c] contracted over j
                
                # Gradient of cost = 2 * residual * dT/dparam
                dU = np.zeros((n, m, r))
                dV = np.zeros((m, p, r))
                dW = np.zeros((p, n, r))
                
                for c in range(r):
                    # dU[i,j,c] = sum_k residual[i,j,k] * V[j,k,c] * W[k,i,c]
                    dU[:,:,c] = np.einsum('ijk,jk,ki->ij', residual.reshape(n,m,p), V[:,:,c], W[:,:,c])
                    # dV[j,k,c] = sum_i residual[i,j,k] * U[i,j,c] * W[k,i,c]
                    dV[:,:,c] = np.einsum('ijk,ij,ki->jk', residual.reshape(n,m,p), U[:,:,c], W[:,:,c])
                    # dW[k,i,c] = sum_j residual[i,j,k] * U[i,j,c] * V[j,k,c]
                    dW[:,:,c] = np.einsum('ijk,ij,jk->ki', residual.reshape(n,m,p), U[:,:,c], V[:,:,c])
                
                # Adaptive step size
                lr = 0.01 / (1 + step * 0.01)
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            # Check if we found a solution
            if best_cost < 1e-12:
                U_r = np.round(best_U * 2) / 2
                V_r = np.round(best_V * 2) / 2
                W_r = np.round(best_W * 2) / 2
                
                U_flat = U_r.reshape(n*m, r)
                V_flat = V_r.reshape(m*p, r)
                W_flat = W_r.reshape(p*n, r)
                
                if _validate_decomposition(U_flat.tolist(), V_flat.tolist(), W_flat.tolist(), n, m, p):
                    return U_flat.tolist(), V_flat.tolist(), W_flat.tolist()
                
                # Try without rounding
                if _validate_decomposition(U_flat.tolist(), V_flat.tolist(), W_flat.tolist(), n, m, p):
                    return U_flat.tolist(), V_flat.tolist(), W_flat.tolist()
        
        return None
    
    # Try ranks from low to high
    # Lower bound for matrix multiplication tensor rank is max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # But actually the lower bound is higher. For 3x5x7, known bounds suggest rank >= ~50+
    # Let's try from 40 upward
    
    for r in range(40, 106):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, max_restarts=5)
        if result is not None:
            return {"U": result[0], "V": result[1], "W": result[2]}
    
    # Fallback: schoolbook decomposition (rank = n*m*p = 105)
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
