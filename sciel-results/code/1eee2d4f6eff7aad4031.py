import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = 3, 3, 4
    # Target tensor T of shape (9, 12, 12)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def compute_tensor(U, V, W):
        # U: (9, r), V: (12, r), W: (12, r)
        return np.einsum('ic,jc,kc->ijk', U, V, W)

    def objective(params, rank):
        U = params[:9*rank].reshape(9, rank)
        V = params[9*rank:9*rank+12*rank].reshape(12, rank)
        W = params[9*rank+12*rank:].reshape(12, rank)
        T_comp = compute_tensor(U, V, W)
        error = np.sum((T_comp - T)**2)
        return error

    start_time = time.time()
    deadline = 110.0 # Leave 10s margin

    best_rank = 36
    best_solution = None

    # Try ranks from 25 to 30
    for rank in range(25, 31):
        if time.time() - start_time > deadline:
            break
            
        found = False
        # Try a few random restarts
        for restart in range(5):
            if time.time() - start_time > deadline:
                break
                
            np.random.seed(rank * 100 + restart)
            # Initialize with small random values
            U = np.random.randn(9, rank) * 0.1
            V = np.random.randn(12, rank) * 0.1
            W = np.random.randn(12, rank) * 0.1
            
            # Flatten for optimizer
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(objective, x0, args=(rank,), method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
                if res.fun < 1e-6:
                    U_opt = res.x[:9*rank].reshape(9, rank)
                    V_opt = res.x[9*rank:9*rank+12*rank].reshape(12, rank)
                    W_opt = res.x[9*rank+12*rank:].reshape(12, rank)
                    
                    # Verify
                    T_comp = compute_tensor(U_opt, V_opt, W_opt)
                    if np.max(np.abs(T_comp - T)) < 1e-5:
                        best_rank = rank
                        best_solution = {
                            "U": U_opt.tolist(),
                            "V": V_opt.tolist(),
                            "W": W_opt.tolist()
                        }
                        found = True
                        break
            except Exception:
                pass

        if found:
            break

    if best_solution is None:
        # Fallback to schoolbook rank 36
        rank = 36
        U = np.zeros((9, rank))
        V = np.zeros((12, rank))
        W = np.zeros((12, rank))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(4):
                    U[i*3+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
        best_solution = {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    return best_solution
