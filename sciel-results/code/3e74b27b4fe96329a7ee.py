"""Find rank-minimal exact tensor decomposition for 2x3x10 matrix multiplication."""
import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    start_time = time.time()
    deadline = start_time + 135
    
    # Schoolbook as fallback
    rank_sb = n * m * p  # 60
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    def check_decomposition(U, V, W, r, tol=1e-9):
        for ip in range(n):
            for kp in range(p):
                M = np.zeros((m, m))
                for rr in range(r):
                    u = U[ip*m:(ip+1)*m, rr]
                    v = np.array([V[j*p+kp, rr] for j in range(m)])
                    M += W[kp*n+ip, rr] * np.outer(u, v)
                if np.max(np.abs(M - np.eye(m))) > tol:
                    return False
        return True
    
    def objective_factory(r):
        def objective(params):
            U_f = params[:n*m*r].reshape(n*m, r)
            V_f = params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
            W_f = params[n*m*r+m*p*r:].reshape(p*n, r)
            
            err = 0.0
            for ip in range(n):
                for kp in range(p):
                    M = np.zeros((m, m))
                    for rr in range(r):
                        u = U_f[ip*m:(ip+1)*m, rr]
                        v = np.array([V_f[j*p+kp, rr] for j in range(m)])
                        M += W_f[kp*n+ip, rr] * np.outer(u, v)
                    err += np.sum((M - np.eye(m))**2)
            return err
        return objective
    
    # Search from low rank upward
    for r in range(8, min(60, best_rank)):
        if time.time() > deadline - 10:
            break
        
        found = False
        num_restarts = 5 if r < 20 else 3
        
        for restart in range(num_restarts):
            if time.time() > deadline - 3:
                break
            if found:
                break
                
            np.random.seed(50000 + r * 1000 + restart)
            
            # Try different scales
            scale = 0.5 if restart % 2 == 0 else 1.0
            U = np.random.randn(n*m, r) * scale
            V = np.random.randn(m*p, r) * scale
            W = np.random.randn(p*n, r) * scale
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            obj = objective_factory(r)
            
            result = minimize(obj, x0, method='L-BFGS-B', maxiter=500, ftol=1e-15, gtol=1e-15)
            
            if result.fun < 1e-12:
                U_f = result.x[:n*m*r].reshape(n*m, r)
                V_f = result.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W_f = result.x[n*m*r+m*p*r:].reshape(p*n, r)
                
                if check_decomposition(U_f, V_f, W_f, r):
                    best_U, best_V, best_W = U_f, V_f, W_f
                    best_rank = r
                    found = True
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
