import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 3, 3, 4
    
    for rank in range(20, 37):
        if time.time() > deadline - 20:
            break
        
        for trial in range(200):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(42 + trial)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            converged = False
            for it in range(500):
                if time.time() > deadline - 20:
                    break
                
                # Solve for U given V, W
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, rank))
                        for k in range(p):
                            for c in range(rank):
                                A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        b = np.ones(p)
                        U[i*m+j, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                
                # Solve for V given U, W
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, rank))
                        for i in range(n):
                            for c in range(rank):
                                A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        b = np.ones(n)
                        V[j*p+k, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                
                # Solve for W given U, V
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, rank))
                        for j in range(m):
                            for c in range(rank):
                                A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        b = np.ones(m)
                        W[k*n+i, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                
                # Check convergence
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            max_err = max(max_err, abs(s - 1))
                
                if max_err < 1e-10:
                    converged = True
                    break
            
            if converged:
                U_list = [[float(U[i,c]) for c in range(rank)] for i in range(n*m)]
                V_list = [[float(V[i,c]) for c in range(rank)] for i in range(m*p)]
                W_list = [[float(W[i,c]) for c in range(rank)] for i in range(p*n)]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
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
