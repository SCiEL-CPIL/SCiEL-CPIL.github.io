import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    n, m, p = 2, 5, 5
    nm = n * m  # 10
    mp = m * p  # 25
    pn = p * n  # 10
    n_constraints = n * m * p  # 50
    
    # Build schoolbook decomposition (rank 50)
    def build_schoolbook():
        R = n * m * p
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U.tolist(), V.tolist(), W.tolist()
    
    best_solution = build_schoolbook()
    best_rank = n * m * p
    
    def solve_U(V, W, R):
        U = np.zeros((nm, R))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, R))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                b = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    U[i*m+j, :] = np.linalg.lstsq(A.T, b, rcond=None)[0]
        return U

    def solve_V(U, W, R):
        V = np.zeros((mp, R))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, R))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                b = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    V[j*p+k, :] = np.linalg.lstsq(A.T, b, rcond=None)[0]
        return V

    def solve_W(U, V, R):
        W = np.zeros((pn, R))
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, R))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                b = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    W[k*n+i, :] = np.linalg.lstsq(A.T, b, rcond=None)[0]
        return W

    def check_error(U, V, W):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err = abs(total - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def try_rank(R, n_restarts=5, max_iter=100, tol=1e-9):
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                return None
            
            # Initialization
            if restart == 0:
                U0, V0, W0 = build_schoolbook()
                U0 = np.array(U0)
                V0 = np.array(V0)
                W0 = np.array(W0)
                np.random.seed(42 + restart)
                if R < n * m * p:
                    idx = np.random.choice(n*m*p, size=R, replace=False)
                    U = U0[:, idx]
                    V = V0[:, idx]
                    W = W0[:, idx]
                else:
                    U = U0.copy()
                    V = V0.copy()
                    W = W0.copy()
            else:
                np.random.seed(100 + restart)
                U = np.random.randn(nm, R) * 0.5
                V = np.random.randn(mp, R) * 0.5
                W = np.random.randn(pn, R) * 0.5
            
            prev_err = np.inf
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    return None
                
                U = solve_U(V, W, R)
                V = solve_V(U, W, R)
                W = solve_W(U, V, R)
                
                err = check_error(U, V, W)
                
                if err < tol:
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    err_rounded = check_error(U_rounded, V_rounded, W_rounded)
                    if err_rounded < tol:
                        return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
                    else:
                        return U.tolist(), V.tolist(), W.tolist()
                
                if abs(prev_err - err) < 1e-12:
                    break
                prev_err = err
        
        return None

    # Try ranks from 25 upwards
    for R in range(25, 50):
        if time.time() > deadline - 10:
            break
        
        # Use more restarts for lower ranks
        restarts = 10 if R < 30 else 5
        result = try_rank(R, n_restarts=restarts)
        if result is not None:
            best_rank = R
            best_solution = result
            break

    return {"U": best_solution[0], "V": best_solution[1], "W": best_solution[2]}
