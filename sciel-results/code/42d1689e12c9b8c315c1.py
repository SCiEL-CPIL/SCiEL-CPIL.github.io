import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = 2, 5, 5
    nm = n * m
    mp = m * p
    pn = p * n
    
    def check_error(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        R = U.shape[1]
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        
        max_err = 0.0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        # Compute sum_r W[k,i,r] U[i,j,r] V[jp,k,r]
                        val = np.sum(W_t[k, i, :] * U_t[i, j, :] * V_t[jp, k, :])
                        target = 1.0 if j == jp else 0.0
                        err = abs(val - target)
                        if err > max_err:
                            max_err = err
        return max_err

    def solve_U(V, W, R):
        U = np.zeros((nm, R))
        # Precompute V and W reshaped
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        
        for i in range(n):
            for j in range(m):
                # A is (p*m, R)
                A = np.zeros((p * m, R))
                b = np.zeros(p * m)
                for k in range(p):
                    for jp in range(m):
                        idx = k * m + jp
                        A[idx, :] = W_t[k, i, :] * V_t[jp, k, :]
                        b[idx] = 1.0 if jp == j else 0.0
                
                # Solve A x = b
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    U[i*m+j, :] = np.linalg.lstsq(A.T, b, rcond=None)[0]
        return U

    def solve_V(U, W, R):
        V = np.zeros((mp, R))
        U_t = U.reshape(n, m, R)
        W_t = W.reshape(p, n, R)
        
        for j in range(m):
            for k in range(p):
                A = np.zeros((n * m, R))
                b = np.zeros(n * m)
                for i in range(n):
                    for jp in range(m):
                        idx = i * m + jp
                        A[idx, :] = W_t[k, i, :] * U_t[i, jp, :]
                        b[idx] = 1.0 if jp == j else 0.0
                        
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    V[j*p+k, :] = np.linalg.lstsq(A.T, b, rcond=None)[0]
        return V

    def solve_W(U, V, R):
        W = np.zeros((pn, R))
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        
        for k in range(p):
            for i in range(n):
                A = np.zeros((m * m, R))
                b = np.zeros(m * m)
                for j in range(m):
                    for jp in range(m):
                        idx = j * m + jp
                        A[idx, :] = U_t[i, j, :] * V_t[jp, k, :]
                        b[idx] = 1.0 if j == jp else 0.0
                        
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    W[k*n+i, :] = np.linalg.lstsq(A.T, b, rcond=None)[0]
        return W

    def try_rank(R, n_restarts=20, max_iter=100):
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                return None
            
            np.random.seed(1000 + restart)
            U = np.random.randn(nm, R) * 0.1
            V = np.random.randn(mp, R) * 0.1
            W = np.random.randn(pn, R) * 0.1
            
            prev_err = np.inf
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                U = solve_U(V, W, R)
                V = solve_V(U, W, R)
                W = solve_W(U, V, R)
                
                err = check_error(U, V, W)
                
                if err < 1e-9:
                    return U, V, W
                
                if abs(prev_err - err) < 1e-12:
                    break
                prev_err = err
        return None

    # Try ranks
    ranks_to_try = [50, 45, 40, 35, 30, 25, 20]
    best_solution = None
    best_rank = 50
    
    # Initialize with schoolbook
    U0 = np.zeros((nm, 50))
    V0 = np.zeros((mp, 50))
    W0 = np.zeros((pn, 50))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    best_solution = (U0, V0, W0)
    
    for R in ranks_to_try:
        if time.time() > deadline - 20:
            break
        res = try_rank(R)
        if res is not None:
            best_solution = res
            best_rank = R
    
    U, V, W = best_solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
