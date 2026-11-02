import numpy as np
import time

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Construct the target tensor T
    # T[a, b, c] where a in [0, n*m), b in [0, m*p), c in [0, n*p)
    T = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    best_result = None
    start_time = time.time()
    deadline = start_time + 110

    # Try ranks from 20 to 36
    for R in range(20, 37):
        if time.time() > deadline:
            break
            
        found = False
        # Use multiple random restarts
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(n*p, R)
            
            # Gradient Descent
            lr = 0.05
            for step in range(200):
                if time.time() > deadline:
                    break
                    
                # T_approx = sum_r U_r V_r^T W_r
                # Using einsum for speed
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                
                error = T - T_approx
                err_norm = np.linalg.norm(error)
                
                if err_norm < 1e-8:
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    found = True
                    break
                
                # Gradients
                # dL/dU[a, r] = sum_{b,c} error[a,b,c] V[b,r] W[c,r]
                grad_U = np.einsum('abc,br,cr->ar', error, V, W)
                grad_V = np.einsum('abc,ar,cr->br', error, U, W)
                grad_W = np.einsum('abc,ar,br->cr', error, U, V)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
            if found:
                break
        
        if found:
            break

    if best_result is None:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(n * p)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][column] = 1
                    V[b][column] = 1
                    W[c][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
