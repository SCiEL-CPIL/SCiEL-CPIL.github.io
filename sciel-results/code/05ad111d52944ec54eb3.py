import numpy as np
import time

def construct():
    n, m, p = 3, 3, 8
    I, J, K = n*m, m*p, p*n  # 9, 24, 24
    deadline = time.time() + 105
    
    T = np.zeros((I, J, K))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    T1 = T.reshape(I, J*K)
    T2 = T.transpose(1, 0, 2).reshape(J, I*K)
    T3 = T.transpose(2, 0, 1).reshape(K, I*J)
    
    def schoolbook():
        rank = 72
        U = np.zeros((I, rank))
        V = np.zeros((J, rank))
        W = np.zeros((K, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    def compute_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1)**2
        return err
    
    for rank in range(71, 50, -1):
        if time.time() > deadline - 5:
            break
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            rng = np.random.RandomState(rank * 1000 + restart)
            U = rng.randn(I, rank)
            V = rng.randn(J, rank)
            W = rng.randn(K, rank)
            
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                try:
                    Z = np.einsum('jr,kr->jkr', V, W).reshape(J*K, rank)
                    ZtZ = Z.T @ Z
                    U = T1 @ Z @ np.linalg.inv(ZtZ)
                    
                    Y = np.einsum('ir,kr->ikr', U, W).reshape(I*K, rank)
                    YtY = Y.T @ Y
                    V = T2 @ Y @ np.linalg.inv(YtY)
                    
                    X = np.einsum('ir,jr->ijr', U, V).reshape(I*J, rank)
                    XtX = X.T @ X
                    W = T3 @ X @ np.linalg.inv(XtX)
                except:
                    break
            
            err = compute_error(U, V, W)
            if err < 1e-2 and rank < best_rank:
                best_U, best_V, best_W, best_rank = U, V, W, rank
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
