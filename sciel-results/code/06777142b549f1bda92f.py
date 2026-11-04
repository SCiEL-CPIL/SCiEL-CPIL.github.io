def _try_ranked_decomposition(n, m, p, rank, time_budget=10.0, seed=42):
    start = time.time()
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    
    lr = 0.05
    for step in range(2000):
        if time.time() - start > time_budget:
            break
        
        # Compute residuals and gradients
        residuals = np.zeros((n, m, p))
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i,j] * V[j,k] * W[k,i])
                    res = val - 1.0
                    residuals[i,j,k] = res
                    dU[i,j] += 2 * res * (V[j,k] * W[k,i])
                    dV[j,k] += 2 * res * (U[i,j] * W[k,i])
                    dW[k,i] += 2 * res * (U[i,j] * V[j,k])
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        if np.sum(residuals**2) < 1e-12:
            break
            
    # Try to round to half-integers
    U_r = np.round(U * 2) / 2.0
    V_r = np.round(V * 2) / 2.0
    W_r = np.round(W * 2) / 2.0
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(np.sum(U_r[i,j] * V_r[j,k] * W_r[k,i]) - 1.0) > 1e-6:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if valid:
        return U_r.tolist(), V_r.tolist(), W_r.tolist(), True
        
    return None, None, None, False
