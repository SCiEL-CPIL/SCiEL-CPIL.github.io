import numpy as np
import time

def construct():
    n, m, p = 4, 4, 4
    deadline = time.time() + 140
    
    def check_and_round(U, V, W):
        # Try rounding to half-integers
        for scale in [2, 1]:  # half-integers or integers
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            res = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                        res += (val - 1)**2
            if res < 1e-9:
                return (U_r, V_r, W_r)
        return None
    
    def try_rank(rank, seed):
        np.random.seed(seed)
        U = np.random.randn(n*m, rank) * 0.3
        V = np.random.randn(m*p, rank) * 0.3
        W = np.random.randn(p*n, rank) * 0.3
        
        for it in range(5000):
            if time.time() > deadline:
                return None
            
            # Update U
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    reg = 1e-10
                    A = M.T @ M + reg * np.eye(rank)
                    rhs = M.T @ b
                    try:
                        U[i*m+j, :] = np.linalg.solve(A, rhs)
                    except:
                        U[i*m+j, :] = np.linalg.lstsq(M, b, rcond=None)[0]
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, rank))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    reg = 1e-10
                    A = M.T @ M + reg * np.eye(rank)
                    rhs = M.T @ b
                    try:
                        V[j*p+k, :] = np.linalg.solve(A, rhs)
                    except:
                        V[j*p+k, :] = np.linalg.lstsq(M, b, rcond=None)[0]
            
            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    reg = 1e-10
                    A = M.T @ M + reg * np.eye(rank)
                    rhs = M.T @ b
                    try:
                        W[k*n+i, :] = np.linalg.solve(A, rhs)
                    except:
                        W[k*n+i, :] = np.linalg.lstsq(M, b, rcond=None)[0]
            
            if it % 200 == 0:
                res = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            res += (val - 1)**2
                if res < 1e-8:
                    result = check_and_round(U, V, W)
                    if result:
                        return result
        
        return None
    
    # Try rank 49 with multiple seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        result = try_rank(49, seed)
        if result:
            U_r, V_r, W_r = result
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Fallback to schoolbook
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
