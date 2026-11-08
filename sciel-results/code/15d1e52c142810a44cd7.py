import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    N_U = n * m  # 6
    N_V = m * p  # 15
    N_W = p * n  # 10
    
    # Precompute indices for the tensor contraction
    # The tensor T[i, j, k] = 1 for all i, j, k
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
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
    
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(N_U)]
        V = [[0] * rank for _ in range(N_V)]
        W = [[0] * rank for _ in range(N_W)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_solution = get_schoolbook()
    
    # Search for minimal rank
    # Lower bound is max(N_U, N_V, N_W) = 15
    # Upper bound is schoolbook rank 30
    for R in range(15, n * m * p):
        if time.time() > deadline:
            break
            
        # Number of restarts depends on remaining time and rank
        # Higher ranks are harder, but we are going up from lower bound
        # Let's allocate time per rank
        remaining_time = deadline - time.time()
        ranks_left = (n * m * p) - R
        time_per_rank = remaining_time / (ranks_left + 1)
        
        # Estimate restarts based on time_per_rank
        # Assume each BFGS run takes ~0.1-0.5s
        max_restarts = int(time_per_rank / 0.2)
        max_restarts = max(1, min(50, max_restarts))
        
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Scale matters. Values around 1 are good.
            np.random.seed(42 + R * 1000 + restart)
            U_init = np.random.randn(N_U, R)
            V_init = np.random.randn(N_V, R)
            W_init = np.random.randn(N_W, R)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def loss(x):
                U = x[:N_U * R].reshape(N_U, R)
                V = x[N_U * R : N_U * R + N_V * R].reshape(N_V, R)
                W = x[N_U * R + N_V * R :].reshape(N_W, R)
                
                # Compute contraction
                # sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                terms = U[u_indices] * V[v_indices] * W[w_indices]
                sums = terms.sum(axis=1)
                
                errors = sums - 1.0
                return np.sum(errors ** 2)
            
            try:
                # BFGS is good for small-medium problems
                res = minimize(loss, x0, method='BFGS', options={'maxiter': 100, 'gtol': 1e-6})
            except Exception:
                continue
                
            if res.fun < 1e-4:
                # Try to snap to half-integers
                x_opt = res.x
                U_opt = x_opt[:N_U * R].reshape(N_U, R)
                V_opt = x_opt[N_U * R : N_U * R + N_V * R].reshape(N_V, R)
                W_opt = x_opt[N_U * R + N_V * R :].reshape(N_W, R)
                
                # Snap to nearest 0.5
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                # Verify
                terms_snap = U_snap[u_indices] * V_snap[v_indices] * W_snap[w_indices]
                sums_snap = terms_snap.sum(axis=1)
                
                if np.allclose(sums_snap, 1.0, atol=1e-9):
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
        
    return best_solution
