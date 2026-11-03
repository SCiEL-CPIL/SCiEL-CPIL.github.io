import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin
    
    # Try ranks from low to high
    for rank in range(50, 73):
        if time.time() > deadline - 10:
            break
        
        best_result = None
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            np.random.seed(rank * 1000 + attempt)
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # Gradient descent
            lr = 0.01
            for it in range(5000):
                if time.time() > deadline - 3:
                    break
                
                # Compute residuals
                residuals = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i*m + j
                            idx_v = j*p + k
                            idx_w = k*n + i
                            residuals[i,j,k] = np.dot(U[idx_u], V[idx_v] * W[idx_w]) - 1
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i*m + j
                            idx_v = j*p + k
                            idx_w = k*n + i
                            res = residuals[i,j,k]
                            grad_U[idx_u] += res * V[idx_v] * W[idx_w]
                            grad_V[idx_v] += res * U[idx_u] * W[idx_w]
                            grad_W[idx_w] += res * U[idx_u] * V[idx_v]
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if np.max(np.abs(residuals)) < 1e-8:
                    break
            
            if np.max(np.abs(residuals)) < 1e-6:
                # Convert to integer/half-integer
                # Round to nearest 0.5
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Verify
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i*m + j
                            idx_v = j*p + k
                            idx_w = k*n + i
                            val = np.dot(U_int[idx_u], V_int[idx_v] * W_int[idx_w])
                            max_err = max(max_err, abs(val - 1))
                
                if max_err < 1e-9:
                    # Convert to lists
                    U_list = [[float(x) for x in row] for row in U_int]
                    V_list = [[float(x) for x in row] for row in V_int]
                    W_list = [[float(x) for x in row] for row in W_int]
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
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
