def _find_rank_233(rank, max_attempts=20, max_iters=500):
    n, m, p = 2, 3, 3
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for attempt in range(max_attempts):
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        lr = 0.01
        for it in range(max_iters):
            T = np.zeros((n, m, p))
            for r in range(rank):
                u = U[:, r].reshape(n, m)
                v = V[:, r].reshape(m, p)
                w = W[:, r].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
            
            err = np.sum((T - 1.0)**2)
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-12:
                break
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for r in range(rank):
                u = U[:, r].reshape(n, m)
                v = V[:, r].reshape(m, p)
                w = W[:, r].reshape(p, n)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            gt = 2.0 * (T[i, j, k] - 1.0)
                            dU[i * m + j, r] += gt * v[j * p + k, r] * w[k * n + i, r]
                            dV[j * p + k, r] += gt * u[i * m + j, r] * w[k * n + i, r]
                            dW[k * n + i, r] += gt * u[i * m + j, r] * v[j * p + k, r]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if it % 50 == 0:
                lr *= 0.95
        
        if best_err < 1e-12:
            # Check if close to half-integers
            close = True
            for mat in [best_U, best_V, best_W]:
                for val in mat:
                    if abs(2 * val - round(2 * val)) > 1e-6:
                        close = False
                        break
                if not close:
                    break
            
            if close:
                U_list = [[round(2 * x) / 2.0 for x in row] for row in best_U]
                V_list = [[round(2 * x) / 2.0 for x in row] for row in best_V]
                W_list = [[round(2 * x) / 2.0 for x in row] for row in best_W]
                
                if _verify_decomposition(U_list, V_list, W_list, n, m, p, rank):
                    return U_list, V_list, W_list
            else:
                # Try snapping anyway
                U_list = [[round(2 * x) / 2.0 for x in row] for row in best_U]
                V_list = [[round(2 * x) / 2.0 for x in row] for row in best_V]
                W_list = [[round(2 * x) / 2.0 for x in row] for row in best_W]
                
                if _verify_decomposition(U_list, V_list, W_list, n, m, p, rank):
                    return U_list, V_list, W_list

    return None
