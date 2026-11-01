import numpy as np
import time
import scipy.optimize as so

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Precompute indices for fast residual calculation
    # We need to map (i, j, k) to indices u, v, w
    u_indices = []
    v_indices = []
    w_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_indices.append(i * m + j)
                v_indices.append(j * p + k)
                w_indices.append(k * n + i)
                
    u_indices = np.array(u_indices)
    v_indices = np.array(v_indices)
    w_indices = np.array(w_indices)
    
    def get_residuals(x):
        # x is a flat vector of length (n*m + m*p + p*n) * R
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        U = x[:size_u * R].reshape(size_u, R)
        V = x[size_u * R : size_u * R + size_v * R].reshape(size_v, R)
        W = x[size_u * R + size_v * R :].reshape(size_w, R)
        
        # Compute product sums
        # prod[r, idx] = U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
        # We want sum over r for each idx
        # Vectorized:
        # U_sel = U[u_indices, :] -> shape (18, R)
        # V_sel = V[v_indices, :] -> shape (18, R)
        # W_sel = W[w_indices, :] -> shape (18, R)
        
        U_sel = U[u_indices, :]
        V_sel = V[v_indices, :]
        W_sel = W[w_indices, :]
        
        sums = np.sum(U_sel * V_sel * W_sel, axis=1)
        return sums - 1.0

    def verify_solution(U, V, W):
        # Check if U, V, W are half-integers
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-8): return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-8): return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-8): return False
        
        # Check exact decomposition
        # Use the rounded values for verification to be safe
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Reconstruct residuals with rounded values
        U_sel = U_r[u_indices, :]
        V_sel = V_r[v_indices, :]
        W_sel = W_r[w_indices, :]
        sums = np.sum(U_sel * V_sel * W_sel, axis=1)
        
        if not np.allclose(sums, 1.0, atol=1e-8):
            return False
        return True

    def solve_rank(R, max_restarts=50):
        size_u = n * m
        size_v = m * p
        size_w = p * n
        total_vars = (size_u + size_v + size_w) * R
        
        # Try multiple random restarts
        for restart in range(max_restarts):
            # Random initialization
            # Scale is important. Schoolbook uses 1s. 
            # Random integers in [-2, 2] * 0.5 might be good.
            x0 = np.random.randint(-4, 5, total_vars) * 0.5
            
            # Sometimes 0s cause issues with gradients if not careful, but LM handles it.
            # Ensure not all zeros for a rank component? Not strictly necessary for LM.
            
            try:
                res = so.least_squares(get_residuals, x0, method='lm', max_nfev=2000, ftol=1e-15, xtol=1e-15, gtol=1e-15)
                x_opt = res.x
                
                # Reshape to check
                U_opt = x_opt[:size_u * R].reshape(size_u, R)
                V_opt = x_opt[size_u * R : size_u * R + size_v * R].reshape(size_v, R)
                W_opt = x_opt[size_u * R + size_v * R :].reshape(size_w, R)
                
                # Check if close to half-integers
                # If the optimization converged to a real solution, it might be close to a half-integer one.
                # We round and verify.
                if verify_solution(U_opt, V_opt, W_opt):
                    return U_opt, V_opt, W_opt
                    
            except Exception:
                continue
                
        return None

    start_time = time.time()
    deadline = 140 # Safety margin for 150s limit

    # Try ranks from 14 up to schoolbook
    for R in [14, 15, 16, 17, 18]:
        if time.time() > deadline:
            break
            
        # Increase restarts for lower ranks? Or keep constant.
        # 50 restarts of LM on ~300 vars should be fast (< 1s each).
        # So 50 restarts * 5 ranks = 250 solves. Should fit in time.
        
        res = solve_rank(R, max_restarts=100)
        if res is not None:
            U, V, W = res
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
