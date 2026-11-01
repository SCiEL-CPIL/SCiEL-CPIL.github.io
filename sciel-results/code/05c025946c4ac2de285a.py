import numpy as np
import time
import itertools

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
    
    def error_function(U, V, W):
        # Vectorized error calculation
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # For each constraint (u, v, w), compute dot(U[u,:], V[v,:]*W[w,:])
        # This is slow in a loop. 
        # Alternative: Build a tensor of products?
        # r is small (20-24). num_constraints is 24.
        # Loop is fine.
        err = 0.0
        for u_idx, v_idx, w_idx in constraints:
            val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
            err += (val - 1.0) ** 2
        return err

    def try_rank(r, time_limit=10.0):
        np.random.seed(42)
        # Try a few random restarts
        for attempt in range(20):
            if time.time() - start_time > timeout - 5.0:
                return None
            
            # Random initialization
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # Gradient Descent
            lr = 0.05
            for step in range(500):
                if time.time() - start_time > timeout - 5.0:
                    return None
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for u_idx, v_idx, w_idx in constraints:
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    diff = (val - 1.0)
                    
                    dU[u_idx, :] += 2 * diff * V[v_idx, :] * W[w_idx, :]
                    dV[v_idx, :] += 2 * diff * U[u_idx, :] * W[w_idx, :]
                    dW[w_idx, :] += 2 * diff * U[u_idx, :] * V[v_idx, :]
                    
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Early stopping check
                if step % 50 == 0:
                    err = error_function(U, V, W)
                    if err < 1e-6:
                        # Round to nearest 0.5
                        U_rounded = np.round(U * 2) / 2
                        V_rounded = np.round(V * 2) / 2
                        W_rounded = np.round(W * 2) / 2
                        
                        # Verify exactness
                        if error_function(U_rounded, V_rounded, W_rounded) < 1e-9:
                            return U_rounded, V_rounded, W_rounded
                        break
        return None

    # Try lower ranks
    for r in [23, 22, 21, 20]:
        sol = try_rank(r, time_limit=10.0)
        if sol:
            best_U, best_V, best_W = sol
            best_rank = r
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
