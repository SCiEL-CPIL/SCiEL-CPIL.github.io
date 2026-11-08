"""Rewritten program for 3x3 x 3x6 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _check_decomposition(U, V, W, n, m, p):
    """Check if U, V, W form a valid decomposition. Returns max residual."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    max_res = 0.0
    for i in range(n):
        for k in range(p):
            # Compute T[a,b,c,d] = sum_r W[k*n+i,r] * U[a*m+b,r] * V[c*p+d,r]
            T = np.zeros((n, m, m, p))
            for r in range(rank):
                w_val = W[k * n + i, r]
                if abs(w_val) < 1e-15:
                    continue
                u_vec = U[:, r].reshape(n, m)
                v_vec = V[:, r].reshape(m, p)
                T += w_val * np.einsum('ab,cd->abcd', u_vec, v_vec)
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            expected = 1.0 if (a == i and b == c and d == k) else 0.0
                            res = abs(T[a, b, c, d] - expected)
                            if res > max_res:
                                max_res = res
    return max_res

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find lower rank decompositions
    # Use a greedy approach: try to eliminate columns from schoolbook
    for target_rank in range(best_rank - 1, max(0, best_rank - 5), -1):
        if time.time() > deadline - 40:
            break
        
        found = False
        # Try random subsets of schoolbook columns
        for attempt in range(100):
            if time.time() > deadline - 10:
                break
            if found:
                break
            
            # Select target_rank columns from the 54 schoolbook columns
            if target_rank >= best_rank:
                cols = list(range(best_rank))
            else:
                cols = list(np.random.choice(best_rank, target_rank, replace=False))
                np.random.shuffle(cols)
            
            U_sub = U[:, cols]
            V_sub = V[:, cols]
            W_sub = W[:, cols]
            
            # Check if this subset works (it won't in general, but let's try)
            res = _check_decomposition(U_sub, V_sub, W_sub, n, m, p)
            if res < 1e-8:
                best_rank = target_rank
                best_result = {"U": U_sub.tolist(), "V": V_sub.tolist(), "W": W_sub.tolist()}
                found = True
                break
        
        # Also try numerical optimization from a perturbed schoolbook
        if not found and time.time() < deadline - 30:
            try:
                from scipy.optimize import minimize
                
                # Parameterize: flatten U, V, W
                U_init, V_init, W_init = _schoolbook(n, m, p)
                # Take first target_rank columns and perturb
                U0 = U_init[:, :target_rank] + 0.01 * np.random.randn(n*m, target_rank)
                V0 = V_init[:, :target_rank] + 0.01 * np.random.randn(m*p, target_rank)
                W0 = W_init[:, :target_rank] + 0.01 * np.random.randn(p*n, target_rank)
                
                x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
                
                def objective(x):
                    U_c = x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_c = x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_c = x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    return _check_decomposition(U_c, V_c, W_c, n, m, p)
                
                res_opt = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
                
                if res_opt.fun < 1e-8:
                    U_opt = res_opt.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_opt = res_opt.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_opt = res_opt.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    final_res = _check_decomposition(U_opt, V_opt, W_opt, n, m, p)
                    if final_res < 1e-8:
                        best_rank = target_rank
                        best_result = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
            except Exception:
                pass
    
    return best_result
