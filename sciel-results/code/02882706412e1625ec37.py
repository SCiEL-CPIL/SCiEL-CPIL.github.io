import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    best_rank = n * m * p
    best_U = best_V = best_W = None
    
    for rank in range(36, best_rank + 1):
        if time.time() - start_time > 130:
            break
        
        found = False
        for attempt in range(50):
            if time.time() - start_time > 130:
                break
            
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            for iteration in range(1000):
                if time.time() - start_time > 130:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # M[k, c] = V[j*p+k, c] * W[k*n+i, c] for k in 0..p-1, c in 0..rank-1
                        M = np.zeros((p, rank))
                        for k in range(p):
                            M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        b = np.ones(p)
                        U[i*m+j, :] = np.linalg.lstsq(M, b, rcond=None)[0]
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        # M[i, c] = U[i*m+j, c] * W[k*n+i, c] for i in 0..n-1, c in 0..rank-1
                        M = np.zeros((n, rank))
                        for i in range(n):
                            M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        b = np.ones(n)
                        V[j*p+k, :] = np.linalg.lstsq(M, b, rcond=None)[0]
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        # M[j, c] = U[i*m+j, c] * V[j*p+k, c] for j in 0..m-1, c in 0..rank-1
                        M = np.zeros((m, rank))
                        for j in range(m):
                            M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        b = np.ones(m)
                        W[k*n+i, :] = np.linalg.lstsq(M, b, rcond=None)[0]
                
                # Check error
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            error += (val - 1.0) ** 2
                
                if error < 1e-12:
                    found = True
                    break
            
            if found:
                # Try to convert to half-integers
                for scale in [2, 1, 4]:
                    U_try = np.round(U * scale) / scale
                    V_try = np.round(V * scale) / scale
                    W_try = np.round(W * scale) / scale
                    
                    error = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_try[i*m+j, :], V_try[j*p+k, :] * W_try[k*n+i, :])
                                error += (val - 1.0) ** 2
                    
                    if error < 1e-8:
                        return {
                            "U": U_try.tolist(),
                            "V": V_try.tolist(),
                            "W": W_try.tolist()
                        }
        
        if found:
            best_rank = rank
            best_U, best_V, best_W = U, V, W
    
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
