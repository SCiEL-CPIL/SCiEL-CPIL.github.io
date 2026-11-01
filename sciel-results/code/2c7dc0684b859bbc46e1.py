def _find_m223_rank10():
    """Find rank-10 decomposition for M_{2,2,3}."""
    import numpy as np
    import time
    
    start = time.time()
    deadline = start + 10  # 10 second budget
    
    n, m, p = 2, 2, 3
    U_dim = n * m  # 4
    V_dim = m * p  # 6
    W_dim = p * n  # 6
    rank = 10
    
    # Target tensor: T[i,j,k] = 1 for all i,j,k
    # Constraint: sum_r U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i] = 1
    
    # Initialize with Strassen (7 terms) + 3 random terms
    np.random.seed(42)
    
    # Strassen coefficients (from current program)
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V_strassen = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # Extend Strassen to 2x2x3: pad V and W with zeros for k=2
    U = np.zeros((rank, U_dim))
    V = np.zeros((rank, V_dim))
    W = np.zeros((rank, W_dim))
    
    # Fill first 7 terms with Strassen
    U[:7, :] = U_strassen.T  # U[r, i*m+j] = U_strassen[i*m+j, r]
    # V_strassen[j*p2+k, r] where p2=2, need to map to j*p3+k where p3=3
    for j in range(2):
        for k in range(2):
            V[:7, j*3+k] = V_strassen[j*2+k, :]
    # W_strassen[k*n+i, r] maps directly since n=2
    W[:7, :] = W_strassen.T
    
    # Initialize last 3 terms randomly
    U[7:, :] = np.random.randn(3, U_dim) * 0.1
    V[7:, :] = np.random.randn(3, V_dim) * 0.1
    W[7:, :] = np.random.randn(3, W_dim) * 0.1
    
    # Gradient descent
    lr = 0.01
    for it in range(5000):
        if time.time() > deadline:
            break
        
        # Compute residuals
        residuals = np.zeros((2, 2, 3))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    residuals[i,j,k] = np.sum(U[:, i*m+j] * V[:, j*p+k] * W[:, k*n+i]) - 1
        
        # Compute gradients (approximate)
        # d/dU[r, im+j] = sum_k V[r, jp+k] * W[r, kn+i] * residual[i,j,k]
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for r in range(rank):
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        dU[r, i*m+j] += V[r, j*p+k] * W[r, k*n+i] * residuals[i,j,k]
                        dV[r, j*p+k] += U[r, i*m+j] * W[r, k*n+i] * residuals[i,j,k]
                        dW[r, k*n+i] += U[r, i*m+j] * V[r, j*p+k] * residuals[i,j,k]
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Check convergence
        if np.max(np.abs(residuals)) < 1e-8:
            break
    
    # Check if solution is valid
    max_err = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                val = np.sum(U[:, i*m+j] * V[:, j*p+k] * W[:, k*n+i])
                max_err = max(max_err, abs(val - 1))
    
    if max_err < 1e-6:
        # Try to round to integers/half-integers
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        max_err_round = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    val = np.sum(U_round[:, i*m+j] * V_round[:, j*p+k] * W_round[:, k*n+i])
                    max_err_round = max(max_err_round, abs(val - 1))
        
        if max_err_round < 1e-9:
            return U_round.tolist(), V_round.tolist(), W_round.tolist()
    
    return None
