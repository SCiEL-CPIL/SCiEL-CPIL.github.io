import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 9
    pn = p * n  # 6
    
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook solution (rank 18)
    rank_sb = n * m * p  # 18
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Try to find lower rank solutions
    for target_rank in range(14, 18):
        if time.time() > deadline - 30:
            break
        
        found = False
        np.random.seed(42)
        
        for attempt in range(20):
            if time.time() > deadline - 20:
                break
            
            # Initialize
            U = np.random.randn(nm, target_rank) * 0.3
            V = np.random.randn(mp, target_rank) * 0.3
            W = np.random.randn(pn, target_rank) * 0.3
            
            # Gradient descent
            lr = 0.01
            for step in range(1000):
                if time.time() > deadline - 15:
                    break
                
                # Compute error and gradient
                error = 0.0
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Current value
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            err = 1.0 - val
                            error += err * err
                            
                            # Gradients
                            for t in range(target_rank):
                                uvw = U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                                # d/dU
                                grad_U[i*m+j, t] += -2 * err * V[j*p+k, t] * W[k*n+i, t]
                                grad_V[j*p+k, t] += -2 * err * U[i*m+j, t] * W[k*n+i, t]
                                grad_W[k*n+i, t] += -2 * err * U[i*m+j, t] * V[j*p+k, t]
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if error < 1e-10:
                    break
            
            # Check if solution is valid
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        max_err = max(max_err, abs(1.0 - val))
            
            if max_err < 1e-6:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Check rounded solution
                max_err_r = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :])
                            max_err_r = max(max_err_r, abs(1.0 - val))
                
                if max_err_r < 1e-6:
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
