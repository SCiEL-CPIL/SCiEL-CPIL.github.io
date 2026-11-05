import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    nm, mp, pn = n*m, m*p, p*n
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_rank = nm * mp  # 45
    best_UVW = None
    
    for rank in range(15, best_rank + 1):
        if time.time() > deadline - 10:
            break
            
        found = False
        for attempt in range(30):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(attempt * 7919 + rank * 131)
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
            
            lr = 0.005
            for step in range(300):
                if time.time() > deadline - 3:
                    break
                    
                # E[a,b,c] = T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r]
                E = T - np.einsum('ar,br,cr->abc', U, V, W)
                err = np.sum(E**2)
                
                if err < 1e-12:
                    found = True
                    best_rank = rank
                    best_UVW = (U.copy(), V.copy(), W.copy())
                    break
                
                # Gradients
                # dL/dU[a,r] = -2 * sum_{b,c} E[a,b,c] * V[b,r] * W[c,r]
                dU = -2 * np.einsum('abc,br,cr->ar', E, V, W)
                dV = -2 * np.einsum('abc,ar,cr->br', E, U, W)
                dW = -2 * np.einsum('abc,ar,br->cr', E, U, V)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            if found:
                break
        
        if found:
            break
    
    if best_UVW is None:
        # Schoolbook fallback
        U = np.zeros((nm, best_rank))
        V = np.zeros((mp, best_rank))
        W = np.zeros((pn, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U, V, W = best_UVW
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
