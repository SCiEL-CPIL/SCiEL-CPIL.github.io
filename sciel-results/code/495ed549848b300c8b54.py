<<<<<<< SEARCH
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
=======
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _try_decompose(n, m, p, rank, deadline):
    """Try to find a rank decomposition using numerical optimization."""
    np.random.seed(42)
    
    best_residual = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for trial in range(5):
        if time.time() > deadline - 15:
            break
        
        np.random.seed(42 + trial * 7)
        
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        lr = 0.01
        best_local_residual = float('inf')
        
        for it in range(200):
            if time.time() > deadline - 15:
                break
            
            res = _compute_residual(U, V, W, n, m, p)
            residual_norm = np.max(np.abs(res))
            
            if residual_norm < 1e-10:
                return _finalize(U, V, W, n, m, p)
            
            if residual_norm < best_local_residual:
                best_local_residual = residual_norm
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r_val = res[i, j, k]
                        dU[i * m + j, :] += r_val * V[j * p + k, :] * W[k * n + i, :]
                        dV[j * p + k, :] += r_val * U[i * m + j, :] * W[k * n + i, :]
                        dW[k * n + i, :] += r_val * U[i * m + j, :] * V[j * p + k, :]
            
            grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
            if grad_norm > 0:
                effective_lr = lr / (1 + grad_norm)
                U -= effective_lr * dU
                V -= effective_lr * dV
                W -= effective_lr * dW
            
            lr *= 0.999
        
        if best_local_residual < best_residual:
            best_residual = best_local_residual
        
        if best_residual < 1e-8:
            return _finalize(best_U, best_V, best_W, n, m, p)
    
    return None
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
