import numpy as np
from scipy.optimize import minimize
import time

def construct():
    n, m, p = 3, 3, 4
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for col in range(rank):
            i = col // (m * p)
            j = (col // p) % m
            k = col % p
            U[i * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + i, col] = 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    def snap_to_grid(matrix):
        return np.round(np.array(matrix) * 2) / 2.0

    def check_solution(U, V, W):
        r = U.shape[1]
        # U: (9, r), V: (12, r), W: (12, r)
        # M_ij[k, k'] = sum_c U[i*m+j, c] V[j*p+k', c] W[k*n+i, c]
        # We want M_ij = I_p for all i, j.
        
        # Reshape for easier indexing
        # U_c[i, j] = U[i*m+j, c]
        # V_c[j, k] = V[j*p+k, c]
        # W_c[k, i] = W[k*n+i, c]
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                # Construct M_ij (p x p)
                M = np.zeros((p, p))
                for c in range(r):
                    u_val = U[i * m + j, c]
                    # v_vals: V[j*p + 0..p-1, c]
                    v_vec = V[j * p : j * p + p, c]
                    # w_vals: W[0*n+i, 1*n+i, ..., (p-1)*n+i]
                    # W index is k*n + i. So for k=0..p-1, indices are 0*n+i, 1*n+i...
                    w_vec = np.array([W[k * n + i, c] for k in range(p)])
                    
                    # M[k, k'] += u_val * v_vec[k'] * w_vec[k]
                    # M = u_val * np.outer(w_vec, v_vec)
                    M += u_val * np.outer(w_vec, v_vec)
                
                # Target is I_p
                target = np.eye(p)
                error += np.sum((M - target) ** 2)
        return error

    def error_function(params):
        r = len(params) // (n*m + m*p + p*n)
        U = params[:n*m*r].reshape(n*m, r)
        V = params[n*m*r:2*n*m*r].reshape(m*p, r)
        W = params[2*n*m*r:].reshape(p*n, r)
        return check_solution(U, V, W)

    start_time = time.time()
    timeout = 140 # Leave 10s margin
    
    best_result = None
    best_rank = n * m * p
    
    # Try ranks from 32 to 35
    for rank in range(32, best_rank + 1):
        if time.time() - start_time > timeout:
            break
            
        found = False
        # Try a few restarts
        for restart in range(10):
            if time.time() - start_time > timeout:
                break
                
            # Random init
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(error_function, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
                if res.fun < 1e-8:
                    U_opt = res.x[:n*m*rank].reshape(n*m, rank)
                    V_opt = res.x[n*m*rank:2*n*m*rank].reshape(m*p, rank)
                    W_opt = res.x[2*n*m*rank:].reshape(p*n, rank)
                    
                    U_snap = snap_to_grid(U_opt)
                    V_snap = snap_to_grid(V_opt)
                    W_snap = snap_to_grid(W_opt)
                    
                    if check_solution(U_snap, V_snap, W_snap) < 1e-6:
                        best_result = {'U': U_snap.tolist(), 'V': V_snap.tolist(), 'W': W_snap.tolist()}
                        best_rank = rank
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break

    if best_result:
        return best_result
    else:
        return schoolbook()
