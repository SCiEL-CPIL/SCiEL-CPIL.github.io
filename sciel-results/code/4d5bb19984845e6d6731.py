import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 3, 5, 6
    R_school = n * m * p  # 90
    
    def als_decompose(R, max_iter=100, n_restarts=30):
        """
        Try to find rank-R decomposition using ALS.
        Returns (U, V, W) as numpy arrays if successful, else None.
        """
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random init
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            prev_err = float('inf')
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # For each k: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                        # So: b[k] = 1 = sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r])
                        # Let M[k,r] = V[j*p+k,r] * W[k*n+i,r]
                        # Solve M @ U[i*m+j,:]^T = b where b = ones(p)
                        W_rows = np.array([k*n + i for k in range(p)])
                        M = V[j*p:(j+1)*p, :] * W[W_rows, :]  # (p, R)
                        b = np.ones(p)
                        U[i*m+j, :] = np.linalg.lstsq(M, b, rcond=None)[0]
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                        # b[i] = 1 = sum_r V[j*p+k,r] * (U[i*m+j,r] * W[k*n+i,r])
                        # M[i,r] = U[i*m+j,r] * W[k*n+i,r]
                        U_rows = np.array([i*m + j for i in range(n)])
                        W_rows = np.array([k*n + i for i in range(n)])
                        M = U[U_rows, :] * W[W_rows, :]  # (n, R)
                        b = np.ones(n)
                        V[j*p+k, :] = np.linalg.lstsq(M, b, rcond=None)[0]
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                        # b[j] = 1 = sum_r W[k*n+i,r] * (U[i*m+j,r] * V[j*p+k,r])
                        # M[j,r] = U[i*m+j,r] * V[j*p+k,r]
                        U_rows = np.array([i*m + j for j in range(m)])
                        V_rows = np.array([j*p + k for j in range(m)])
                        M = U[U_rows, :] * V[V_rows, :]  # (m, R)
                        b = np.ones(m)
                        W[k*n+i, :] = np.linalg.lstsq(M, b, rcond=None)[0]
                
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err = max(err, abs(val - 1.0))
                
                if err < 1e-10:
                    # Check convergence
                    if abs(prev_err - err) < 1e-12:
                        return U, V, W
                prev_err = err
            
            # Check final error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err = max(err, abs(val - 1.0))
            if err < 1e-8:
                return U, V, W
        
        return None
    
    # Try ranks from 20 to 90
    for R in range(20, R_school + 1):
        if time.time() > deadline - 15:
            break
        result = als_decompose(R)
        if result is not None:
            U, V, W = result
            # Try to rationalize
            def rationalize_matrix(arr, max_denom=50):
                arr_r = np.zeros_like(arr)
                for idx in np.ndindex(arr.shape):
                    val = arr[idx]
                    best = (0, 1)
                    for den in range(1, max_denom + 1):
                        num = round(val * den)
                        if abs(val - num / den) < 1e-9:
                            if den < best[1] or (den == best[1] and abs(num) < abs(best[0])):
                                best = (num, den)
                            if den == 1:
                                break
                    arr_r[idx] = best[0] / best[1]
                return arr_r
            
            U_r = rationalize_matrix(U, 50)
            V_r = rationalize_matrix(V, 50)
            W_r = rationalize_matrix(W, 50)
            
            # Verify rationalized
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                        err = max(err, abs(val - 1.0))
            
            if err < 1e-8:
                return {'U': U_r.tolist(), 'V': V_r.tolist(), 'W': W_r.tolist()}
            else:
                # Return floating point (should be close enough)
                return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Fallback: schoolbook
    R = R_school
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
