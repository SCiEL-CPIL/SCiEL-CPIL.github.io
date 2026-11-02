import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Seed solution (Rank 24)
    rank_seed = n * m * p
    U_seed = np.zeros((n * m, rank_seed), dtype=np.float64)
    V_seed = np.zeros((m * p, rank_seed), dtype=np.float64)
    W_seed = np.zeros((p * n, rank_seed), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j, col] = 1
                V_seed[j * p + k, col] = 1
                W_seed[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_seed, V_seed, W_seed
    best_rank = rank_seed
    
    start_time = time.time()
    timeout = 110.0 # Leave 10s margin
    
    # Constraints: (u_idx, v_idx, w_idx)
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                constraints.append((u_idx, v_idx, w_idx))
                
    num_constraints = len(constraints)
    
    def verify(U, V, W, tol=1e-9):
        for u_idx, v_idx, w_idx in constraints:
            val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
            if abs(val - 1.0) > tol:
                return False
        return True

    def round_to_half_int(U, V, W):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r):
            return U_r, V_r, W_r
        return None

    def try_rank(r):
        # Method 1: Sparse random search over half-integer values
        # This is often effective for small tensor ranks where solutions are sparse
        for attempt in range(200):
            if time.time() - start_time > timeout - 5.0:
                return None
            
            np.random.seed(42 + attempt)
            
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            # Values from half-integers, restricted range
            vals = [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]
            
            for c in range(r):
                # Sparse pattern: 1-2 non-zeros per column
                nz_u = np.random.choice(n * m, size=np.random.randint(1, 3), replace=False)
                for idx in nz_u:
                    U[idx, c] = np.random.choice(vals)
                
                nz_v = np.random.choice(m * p, size=np.random.randint(1, 3), replace=False)
                for idx in nz_v:
                    V[idx, c] = np.random.choice(vals)
                
                nz_w = np.random.choice(p * n, size=np.random.randint(1, 3), replace=False)
                for idx in nz_w:
                    W[idx, c] = np.random.choice(vals)
            
            if verify(U, V, W):
                return U, V, W

        # Method 2: Gradient descent / Optimization with multiple restarts
        # Use a vectorized error function for speed
        u_idxs = np.array([c[0] for c in constraints])
        v_idxs = np.array([c[1] for c in constraints])
        w_idxs = np.array([c[2] for c in constraints])
        
        def error_function(params, r):
            U = params[0:n*m*r].reshape(n*m, r)
            V = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
            W = params[n*m*r + m*p*r:].reshape(p*n, r)
            
            # Vectorized computation of dot products
            # U[u_idx, :] dot (V[v_idx, :] * W[w_idx, :])
            # Shape of U[u_idxs, :] is (num_constraints, r)
            # Shape of V[v_idxs, :] * W[w_idxs, :] is (num_constraints, r)
            
            UVW = U[u_idxs, :] * (V[v_idxs, :] * W[w_idxs, :])
            vals = np.sum(UVW, axis=1)
            
            err = np.sum((vals - 1.0) ** 2)
            return err

        for attempt in range(50):
            if time.time() - start_time > timeout - 5.0:
                return None
            
            # Random initialization
            np.random.seed(1000 + attempt)
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use L-BFGS-B
            try:
                res = minimize(error_function, params0, args=(r,), method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            except Exception:
                continue
                
            if res.success or res.fun < 1e-8:
                # Extract U, V, W
                U_opt = res.x[0:n*m*r].reshape(n*m, r)
                V_opt = res.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_opt = res.x[n*m*r + m*p*r:].reshape(p*n, r)
                
                # Round to nearest 0.5
                result = round_to_half_int(U_opt, V_opt, W_opt)
                if result:
                    return result

        return None

    # Try lower ranks
    # Known lower bound for 2x3x4 is around 17-18, but 20-23 is a realistic target for quick search
    # We start from higher ranks to increase probability of finding a solution quickly
    for r in [23, 22, 21, 20]:
        if time.time() - start_time > timeout - 10.0:
            break
        sol = try_rank(r)
        if sol:
            best_U, best_V, best_W = sol
            best_rank = r
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
