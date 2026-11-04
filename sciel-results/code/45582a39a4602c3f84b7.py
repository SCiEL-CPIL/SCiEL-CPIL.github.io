import time
import numpy as np

def construct():
    n, m, p = 4, 5, 6
    deadline = time.time() + 140
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find lower rank using gradient descent
    # For each r, try multiple random initializations
    
    best_rank = n * m * p
    best_result = None
    
    for r in range(119, 80, -1):
        if time.time() > deadline:
            break
        
        # Try 3 random initializations
        for trial in range(3):
            if time.time() > deadline:
                break
            
            np.random.seed(1000 + r * 10 + trial)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Scale to make initial values reasonable
            U *= 0.5
            V *= 0.5
            W *= 0.5
            
            # Gradient descent with adaptive learning rate
            lr = 0.1
            for it in range(300):
                if time.time() > deadline:
                    break
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            ui = U[i*m+j, :]
                            vj = V[j*p+k, :]
                            wk = W[k*n+i, :]
                            val = np.dot(ui * vj, wk) - 1
                            grad_U[i*m+j, :] += 2 * val * vj * wk
                            grad_V[j*p+k, :] += 2 * val * ui * wk
                            grad_W[k*n+i, :] += 2 * val * ui * vj
                
                # Adaptive learning rate
                grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
                if grad_norm > 0:
                    step = lr * (1.0 / (1.0 + grad_norm))
                    U -= step * grad_U
                    V -= step * grad_V
                    W -= step * grad_W
                
                # Check convergence
                if it % 30 == 0:
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U[i*m+j,:] * V[j*p+k,:], W[k*n+i,:])
                                max_err = max(max_err, abs(val - 1))
                    if max_err < 1e-4:
                        break
            
            # Final verification
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j,:] * V[j*p+k,:], W[k*n+i,:])
                        max_err = max(max_err, abs(val - 1))
            
            if max_err < 1e-4:
                best_rank = r
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    return schoolbook()
