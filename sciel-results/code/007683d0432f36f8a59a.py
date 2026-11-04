import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Precompute indices for the tensor T
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    # We need to solve sum_alpha U[u, alpha] V[v, alpha] W[w, alpha] = T[u, v, w]
    
    # List of (u, v, w) where T is 1
    ones = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                ones.append((u, v, w))
                
    # List of (u, v, w) where T is 0
    # We can just check all u, v, w
    # To save time, we only check w that are "plausible" or all of them?
    # All w in [0, p*n)
    zeros = []
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                # Check if (u,v,w) is in ones
                # Efficient check:
                # i = u // m, j = u % m
                # j' = v // p, k = v % p
                # if j == j': w_expected = k * n + i
                # else no match
                i = u // m
                j = u % m
                jp = v // p
                k = v % p
                if j == jp:
                    w_exp = k * n + i
                    if w != w_exp:
                        zeros.append((u, v, w))
                else:
                    zeros.append((u, v, w))
                    
    # We can reduce zeros list size by only checking w != w_exp when j==jp
    # But the list is small enough (9*12*12 = 1296 max)
    
    def loss_and_grad(x, r):
        # x is flattened U, V, W
        # U: 9*r, V: 12*r, W: 12*r
        size_u = n * m * r
        size_v = m * p * r
        size_w = p * n * r
        
        U = x[:size_u].reshape(n * m, r)
        V = x[size_u:size_u+size_v].reshape(m * p, r)
        W = x[size_u+size_v:size_u+size_v+size_w].reshape(p * n, r)
        
        loss = 0.0
        grad_u = np.zeros_like(U)
        grad_v = np.zeros_like(V)
        grad_w = np.zeros_like(W)
        
        # Compute contributions for ones
        for u, v, w in ones:
            # sum_alpha U[u, alpha] V[v, alpha] W[w, alpha]
            val = np.sum(U[u, :] * V[v, :] * W[w, :])
            err = val - 1.0
            loss += err * err
            # Gradient
            # d/dU[u, a] = 2 * err * V[v, a] * W[w, a]
            grad_u[u, :] += 2.0 * err * V[v, :] * W[w, :]
            grad_v[v, :] += 2.0 * err * U[u, :] * W[w, :]
            grad_w[w, :] += 2.0 * err * U[u, :] * V[v, :]
            
        # Compute contributions for zeros
        for u, v, w in zeros:
            val = np.sum(U[u, :] * V[v, :] * W[w, :])
            err = val
            loss += err * err
            grad_u[u, :] += 2.0 * err * V[v, :] * W[w, :]
            grad_v[v, :] += 2.0 * err * U[u, :] * W[w, :]
            grad_w[w, :] += 2.0 * err * U[u, :] * V[v, :]
            
        return loss, np.concatenate([grad_u.flatten(), grad_v.flatten(), grad_w.flatten()])
        
    def check_exact(U, V, W, r):
        # Check if rounding to half-integers gives exact solution
        # Try rounding to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify
        for u, v, w in ones:
            val = np.sum(U_r[u, :] * V_r[v, :] * W_r[w, :])
            if not np.isclose(val, 1.0, atol=1e-9):
                return False
        for u, v, w in zeros:
            val = np.sum(U_r[u, :] * V_r[v, :] * W_r[w, :])
            if not np.isclose(val, 0.0, atol=1e-9):
                return False
        return True
        
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    start_time = time.time()
    max_time = 110 # Safety margin
    
    # Try ranks from 15 to 35
    for r in range(15, 36):
        if time.time() - start_time > max_time:
            break
            
        found = False
        # Try a few random initializations
        for attempt in range(5):
            if time.time() - start_time > max_time:
                break
                
            np.random.seed(attempt * 100 + r)
            # Initialize small random values
            U_init = np.random.randn(n * m, r) * 0.1
            V_init = np.random.randn(m * p, r) * 0.1
            W_init = np.random.randn(p * n, r) * 0.1
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = opt.minimize(
                    lambda x: loss_and_grad(x, r)[0], 
                    x0, 
                    jac=lambda x: loss_and_grad(x, r)[1], 
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success or res.fun < 1e-8:
                    U_res = res.x[:size_u].reshape(n * m, r)
                    V_res = res.x[size_u:size_u+size_v].reshape(m * p, r)
                    W_res = res.x[size_u+size_v:size_u+size_v+size_w].reshape(p * n, r)
                    
                    if check_exact(U_res, V_res, W_res, r):
                        found = True
                        best_rank = r
                        best_U = U_res
                        best_V = V_res
                        best_W = W_res
                        break
            except Exception:
                pass
        
        if found:
            break
            
    # Convert to list of lists for output
    U_out = [[0] * best_rank for _ in range(n * m)]
    V_out = [[0] * best_rank for _ in range(m * p)]
    W_out = [[0] * best_rank for _ in range(p * n)]
    
    for i in range(n * m):
        for j in range(best_rank):
            U_out[i][j] = best_U[i, j]
    for i in range(m * p):
        for j in range(best_rank):
            V_out[i][j] = best_V[i, j]
    for i in range(p * n):
        for j in range(best_rank):
            W_out[i][j] = best_W[i, j]
            
    return {"U": U_out, "V": V_out, "W": W_out}
