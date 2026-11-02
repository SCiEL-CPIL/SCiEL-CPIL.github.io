import numpy as np
import time

def construct():
    n, m, p = 3, 5, 7
    start_time = time.time()
    deadline = 150 - 5  # 5s safety margin
    
    # Schoolbook decomposition
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
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Precompute ones
    ones = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ones.append((i*m+j, j*p+k, k*n+i))
    ones_set = set(ones)
    
    # Zero sample
    import random
    random.seed(42)
    all_entries = [(a,b,c) for a in range(n*m) for b in range(m*p) for c in range(p*n)]
    zeros = [e for e in all_entries if e not in ones_set]
    zeros_sample = random.sample(zeros, min(2000, len(zeros)))
    
    def is_valid(U_r, V_r, W_r, r):
        for (a,b,c) in all_entries:
            val = np.dot(np.multiply(U_r[a], V_r[b]), W_r[c])
            if (a,b,c) in ones_set:
                if abs(val - 1.0) > 1e-6:
                    return False
            else:
                if abs(val) > 1e-6:
                    return False
        return True
    
    # Try to find lower rank
    for r in range(30, 106):
        if time.time() > deadline:
            break
        
        U_r = U[:, :r].copy()
        V_r = V[:, :r].copy()
        W_r = W[:, :r].copy()
        
        lr = 0.001
        for it in range(200):
            if time.time() > deadline:
                break
            
            dU = np.zeros_like(U_r)
            dV = np.zeros_like(V_r)
            dW = np.zeros_like(W_r)
            
            for (a,b,c) in ones:
                uv = U_r[a] * V_r[b]
                val = np.dot(uv, W_r[c])
                err = val - 1.0
                dU[a] += 2 * err * V_r[b] * W_r[c]
                dV[b] += 2 * err * U_r[a] * W_r[c]
                dW[c] += 2 * err * U_r[a] * V_r[b]
            
            for (a,b,c) in zeros_sample:
                uv = U_r[a] * V_r[b]
                val = np.dot(uv, W_r[c])
                dU[a] += 2 * val * V_r[b] * W_r[c]
                dV[b] += 2 * val * U_r[a] * W_r[c]
                dW[c] += 2 * val * U_r[a] * V_r[b]
            
            U_r -= lr * dU
            V_r -= lr * dV
            W_r -= lr * dW
        
        if is_valid(U_r, V_r, W_r, r):
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = r
            break
    
    # Round to half-integers
    U_r = np.round(best_U * 2) / 2
    V_r = np.round(best_V * 2) / 2
    W_r = np.round(best_W * 2) / 2
    
    if not is_valid(U_r, V_r, W_r, best_rank):
        U_r, V_r, W_r = U, V, W
        best_rank = rank
    
    U_list = [[float(x) for x in U_r[i]] for i in range(n*m)]
    V_list = [[float(x) for x in V_r[i]] for i in range(m*p)]
    W_list = [[float(x) for x in W_r[i]] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
