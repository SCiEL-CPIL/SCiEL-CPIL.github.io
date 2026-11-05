import numpy as np
import time
import itertools

def construct():
    n, m, p = 3, 4, 8
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Build the target tensor: T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # Otherwise T[a,b,c] = 0
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                target[(a, b, c)] = 1.0
    
    def compute_error(U, V, W, rank):
        """Compute sum of squared errors over all (a,b,c) triples."""
        error = 0.0
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    val = np.dot(U[a] * V[b], W[c])
                    true_val = target.get((a, b, c), 0.0)
                    error += (val - true_val)**2
        return error
    
    best_rank = n*m*p
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from 96 down to 90
    for rank in range(96, 89, -1):
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(rank * 100 + trial)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Adaptive learning rate gradient descent
            lr = 0.01
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for a in range(n*m):
                    for b in range(m*p):
                        for c in range(p*n):
                            val = np.dot(U[a] * V[b], W[c])
                            true_val = target.get((a, b, c), 0.0)
                            residual = val - true_val
                            if abs(residual) > 1e-10:
                                grad_U[a] += 2 * residual * V[b] * W[c]
                                grad_V[b] += 2 * residual * U[a] * W[c]
                                grad_W[c] += 2 * residual * U[a] * V[b]
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if iteration % 50 == 0:
                    err = compute_error(U, V, W, rank)
                    if err < 1e-8:
                        break
            
            err = compute_error(U, V, W, rank)
            if err < 1e-8:
                # Round to integers
                U_rounded = np.round(U)
                V_rounded = np.round(V)
                W_rounded = np.round(W)
                
                err_rounded = compute_error(U_rounded, V_rounded, W_rounded, rank)
                if err_rounded < 1e-8:
                    best_rank = rank
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    found = True
                    break
        
        if found:
            break
    
    if best_U is None:
        # Schoolbook
        best_rank = n*m*p
        best_U = np.zeros((n*m, best_rank))
        best_V = np.zeros((m*p, best_rank))
        best_W = np.zeros((p*n, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i*m+j, col] = 1
                    best_V[j*p+k, col] = 1
                    best_W[k*n+i, col] = 1
                    col += 1
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
