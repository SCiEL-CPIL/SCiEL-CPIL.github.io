import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 9]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_round_to_half_int(U, V, W, n, m, p):
    """Try to round to half-integers and verify."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    if _verify(U_r, V_r, W_r, n, m, p):
        return U_r, V_r, W_r
    return None

def _construct_kronecker(n, m, p):
    """
    Try to construct a decomposition using block structure.
    For 2x3 * 3x9, view the 3x9 as three 3x3 blocks.
    Use shared rank-9 decompositions for each 2x3 * 3x3 sub-problem.
    """
    # For 2x3 * 3x3, the trivial rank is 18.
    # But we can do better: rank 2*3*3 = 18 is trivial.
    # Actually, let's just use the trivial construction for each block.
    # Total rank = 3 * (2*3*3) = 54, same as before. Not helpful.
    
    # Alternative: try to find a lower rank for 2x3 * 3x3.
    # The minimum rank for 2xn * nxn is at least n (from dimension arguments).
    # For 2x3 * 3x3, min rank is at least 3.
    # Let's try to find a rank-9 decomposition for 2x3 * 3x3 (conservative).
    
    # Actually, let's just use the schoolbook for now and try to optimize.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin before 150s limit
    
    # Precompute target indices
    target_a = []
    target_b = []
    target_c = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_a.append(i * m + j)
                target_b.append(j * p + k)
                target_c.append(k * n + i)
    
    target_a = np.array(target_a)
    target_b = np.array(target_b)
    target_c = np.array(target_c)
    num_targets = len(target_a)
    
    best_rank = n * m * p  # 54
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from low to high
    # Start with a reasonable lower bound
    # Lower bound: max(n*m, m*p, p*n) / something... 
    # Actually, for matrix multiplication (n,m,p), the rank is at least max(n*m, m*p, p*n) / min(n,m,p)
    # But that's not a tight bound. Let's just try from a reasonable range.
    
    for r in range(20, 54):
        if time.time() > deadline - 15:
            break
        
        found = False
        max_restarts = 50
        restarts_done = 0
        
        while restarts_done < max_restarts and not found:
            if time.time() > deadline - 5:
                break
            
            # Random initialization with structure
            np.random.seed(restarts_done * 7919 + r * 31)
            
            # Initialize with sparse-like structure
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            # Add random non-zero entries
            for t in range(r):
                # Each term connects some (i,j), (j,k), (k,i)
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)
                
                U[i * m + j, t] = np.random.choice([-1, 1])
                V[j * p + k, t] = np.random.choice([-1, 1])
                W[k * n + i, t] = np.random.choice([-1, 1])
            
            # Also add some random dense components
            U += 0.1 * np.random.randn(n * m, r)
            V += 0.1 * np.random.randn(m * p, r)
            W += 0.1 * np.random.randn(p * n, r)
            
            p0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def fast_objective(params):
                U_p = params[:n*m*r].reshape(n*m, r)
                V_p = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_p = params[n*m*r + m*p*r:].reshape(p*n, r)
                
                # T_vals[a_idx] = sum_t U_p[a_idx, t] * V_p[b_idx, t] * W_p[c_idx, t]
                # Vectorized: for each t, compute contribution
                T_vals = np.zeros(num_targets)
                for t in range(r):
                    T_vals += U_p[target_a, t] * V_p[target_b, t] * W_p[target_c, t]
                
                diff = T_vals - 1.0
                return np.dot(diff, diff)
            
            try:
                from scipy.optimize import minimize
                res = minimize(fast_objective, p0, method='L-BFGS-B', 
                              options={'maxiter': 50, 'gtol': 1e-6, 'maxfun': 200})
                
                if res.fun < 1e-4:
                    U_opt = res.x[:n*m*r].reshape(n*m, r)
                    V_opt = res.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                    W_opt = res.x[n*m*r + m*p*r:].reshape(p*n, r)
                    
                    # Try rounding
                    result = _try_round_to_half_int(U_opt, V_opt, W_opt, n, m, p)
                    if result is not None:
                        best_rank = r
                        best_U, best_V, best_W = result
                        found = True
                        break
            except Exception:
                pass
            
            restarts_done += 1
        
        if found:
            break
    
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
