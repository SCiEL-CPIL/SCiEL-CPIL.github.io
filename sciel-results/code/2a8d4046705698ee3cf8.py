import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Try to find low-rank decomposition
    for rank in [16, 17, 18, 19, 20]:
        result = find_decomposition(n, m, p, rank)
        if result is not None:
            return result
    # Fallback to trivial
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def find_decomposition(n, m, p, rank):
    nm = n * m
    mp = m * p
    pn = p * n
    
    def compute_error(U, V, W):
        err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for s in range(rank):
                        val += U[i*m+j][s] * V[j*p+k][s] * W[k*n+i][s]
                    err += (1 - val) ** 2
        return err
    
    best_result = None
    best_err = float('inf')
    
    for seed in range(200):
        np.random.seed(seed)
        U = np.random.uniform(-1, 1, size=(nm, rank))
        V = np.random.uniform(-1, 1, size=(mp, rank))
        W = np.random.uniform(-1, 1, size=(pn, rank))
        
        # Gradient descent
        lr = 0.005
        for step in range(2000):
            dU = np.zeros((nm, rank))
            dV = np.zeros((mp, rank))
            dW = np.zeros((pn, rank))
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for s in range(rank):
                            val += U[i*m+j][s] * V[j*p+k][s] * W[k*n+i][s]
                        err = 1 - val
                        for s in range(rank):
                            dU[i*m+j][s] += 2 * err * V[j*p+k][s] * W[k*n+i][s]
                            dV[j*p+k][s] += 2 * err * U[i*m+j][s] * W[k*n+i][s]
                            dW[k*n+i][s] += 2 * err * U[i*m+j][s] * V[j*p+k][s]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        err = compute_error(U.tolist(), V.tolist(), W.tolist())
        if err < best_err:
            best_err = err
            best_result = (U.copy(), V.copy(), W.copy())
        
        if err < 1e-8:
            break
    
    if best_result is None or best_err > 1e-6:
        return None
    
    U, V, W = best_result
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    err_r = compute_error(U_r.tolist(), V_r.tolist(), W_r.tolist())
    if err_r < 1e-6:
        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return None
