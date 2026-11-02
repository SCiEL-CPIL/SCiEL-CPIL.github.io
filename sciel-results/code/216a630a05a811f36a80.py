import numpy as np
import time

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds to be safe
    
    n, m, p = DIMENSIONS
    # T[i,j,k] = 1 for all i,j,k (the matrix multiplication tensor)
    # We need: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,j,k
    
    # Try rank 49 first
    for rank in [49, 50, 55, 60, 64]:
        if time.time() > deadline - 5:
            break
            
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Use gradient descent to minimize sum of (sum_t ... - 1)^2
        lr = 0.01
        for step in range(5000):
            if time.time() > deadline - 5:
                break
                
            # Compute residuals
            residuals = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residuals[i,j,k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]) - 1.0
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res = residuals[i,j,k]
                        grad_U[i*m+j, :] += 2 * res * V[j*p+k, :] * W[k*n+i, :]
                        grad_V[j*p+k, :] += 2 * res * U[i*m+j, :] * W[k*n+i, :]
                        grad_W[k*n+i, :] += 2 * res * U[i*m+j, :] * V[j*p+k, :]
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if step % 500 == 0:
                loss = np.mean(residuals**2)
                if loss < 1e-12:
                    break
        
        # Check if we found a valid solution
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-10:
            # Convert to list of lists with rational approximation
            U_list = [[0]*rank for _ in range(n*m)]
            V_list = [[0]*rank for _ in range(m*p)]
            W_list = [[0]*rank for _ in range(p*n)]
            for i in range(n*m):
                for t in range(rank):
                    U_list[i][t] = float(U[i,t])
            for i in range(m*p):
                for t in range(rank):
                    V_list[i][t] = float(V[i,t])
            for i in range(p*n):
                for t in range(rank):
                    W_list[i][t] = float(W[i,t])
            
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
