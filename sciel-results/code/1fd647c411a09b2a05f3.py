import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 135
    
    def tidx():
        return [(i*m+j, j*p+k, k*n+i) for i in range(n) for j in range(m) for k in range(p)]
    
    idx = tidx()
    idx_set = set(idx)
    
    def verify(U, V, W, r):
        for (a, b, c) in idx:
            s = sum(U[a][cc] * V[b][cc] * W[c][cc] for cc in range(r))
            if abs(s - 1) > 1e-9:
                return False
        for _ in range(500):
            a = np.random.randint(0, n*m)
            b = np.random.randint(0, m*p)
            c = np.random.randint(0, p*n)
            if (a, b, c) in idx_set:
                continue
            s = sum(U[a][cc] * V[b][cc] * W[c][cc] for cc in range(r))
            if abs(s) > 1e-9:
                return False
        return True
    
    # Try random search with gradient descent for various r
    for r in [25, 30, 40, 50, 60, 80, 100]:
        if time.time() > deadline - 10:
            break
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            np.random.seed(seed * 31 + r)
            U = np.random.randn(n*m, r) * 0.3
            V = np.random.randn(m*p, r) * 0.3
            W = np.random.randn(p*n, r) * 0.3
            
            for it in range(300):
                if time.time() > deadline - 3:
                    break
                gU = np.zeros_like(U)
                gV = np.zeros_like(V)
                gW = np.zeros_like(W)
                loss = 0.0
                for (a, b, c) in idx:
                    s = sum(U[a, cc] * V[b, cc] * W[c, cc] for cc in range(r))
                    e = s - 1.0
                    loss += e*e
                    for cc in range(r):
                        gU[a, cc] += 2*e*V[b, cc]*W[c, cc]
                        gV[b, cc] += 2*e*U[a, cc]*W[c, cc]
                        gW[c, cc] += 2*e*U[a, cc]*V[b, cc]
                ng = np.sqrt(np.sum(gU**2)+np.sum(gV**2)+np.sum(gW**2))
                lr = 0.05/(1+ng)
                U -= lr*gU; V -= lr*gV; W -= lr*gW
                if loss < 1e-16:
                    break
            
            if verify(U.tolist(), V.tolist(), W.tolist(), r):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Schoolbook fallback
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
