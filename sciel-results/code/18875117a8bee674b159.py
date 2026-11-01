import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Try ranks from low to high
    best_result = None
    
    for R in range(8, 25):
        if time.time() > deadline - 15:
            break
        
        # Multiple random restarts
        found = False
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + restart)
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
            
            # Use a simple iterative method
            lr = 0.005
            for step in range(10000):
                if time.time() > deadline - 3:
                    break
                
                # Compute gradients
                grad_U = np.zeros((n*m, R))
                grad_V = np.zeros((m*p, R))
                grad_W = np.zeros((p*n, R))
                err = 0.0
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            e = 1.0 - s
                            err += e*e
                            if err > 1e6:
                                break
                            grad_U[i*m+j, :] -= 2*e*V[j*p+k, :] * W[k*n+i, :]
                            grad_V[j*p+k, :] -= 2*e*U[i*m+j, :] * W[k*n+i, :]
                            grad_W[k*n+i, :] -= 2*e*U[i*m+j, :] * V[j*p+k, :]
                
                if err < 1e-14:
                    found = True
                    break
                
                # Adaptive learning rate
                grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
                if grad_norm > 0:
                    step_size = min(lr, 0.1/grad_norm)
                    U -= step_size * grad_U
                    V -= step_size * grad_V
                    W -= step_size * grad_W
            
            if found:
                break
        
        if found:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
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
