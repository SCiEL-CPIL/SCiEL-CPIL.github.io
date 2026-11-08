import numpy as np
import time

def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = 140
    
    # Schoolbook as fallback
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
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank
    
    for r in range(12, 36):
        if time.time() > deadline:
            break
        found = _try_rank_als(n, m, p, r, deadline)
        if found is not None:
            U, V, W = found
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = r
            break
    
    return best_result

def _try_rank_als(n, m, p, r, deadline):
    np.random.seed(42)
    
    for attempt in range(100):
        if time.time() > deadline:
            return None
        
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        for it in range(200):
            if time.time() > deadline:
                return None
            
            # Update U: for each (i,j), solve for U[i*m+j, :]
            # Equation: sum_k' sum_t U[i,j,t] * W[k',i,t] * V[j,k',t] = 1 for all k'
            # This is: sum_t U[i,j,t] * (sum_k' W[k',i,t] * V[j,k',t] * delta(k',k)) ... 
            # Actually for each k: sum_t U[i,j,t] * W[k,i,t] * V[j,k,t] = 1
            # So for fixed (i,j), we have p equations in r unknowns U[i,j,0..r-1]:
            # sum_t U[i,j,t] * (W[k,i,t] * V[j,k,t]) = 1 for k=0..p-1
            # This is a linear system: M_{k,t} = W[k,i,t]*V[j,k,t], solve M * U[i,j,:] = ones_p
            
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = W[k*n+i, :] * V[j*p+k, :]
                    # Solve M * x = ones_p (least squares)
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[i*m+j, :] = x
            
            # Update V: for each (j,k), solve for V[j*p+k, :]
            # Equation: sum_t V[j,k,t] * U[i,j,t] * W[k,i,t] = 1 for all i
            # M_{i,t} = U[i,j,t]*W[k,i,t], solve M * V[j,k,:] = ones_n
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V[j*p+k, :] = x
            
            # Update W: for each (k,i), solve for W[k*n+i, :]
            # Equation: sum_t W[k,i,t] * U[i,j,t] * V[j,k,t] = 1 for all j
            # M_{j,t} = U[i,j,t]*V[j,k,t], solve M * W[k,i,:] = ones_m
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W[k*n+i, :] = x
            
            # Check convergence
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        max_err = max(max_err, abs(val - 1))
            if max_err < 1e-8:
                return U, V, W
    
    return None
