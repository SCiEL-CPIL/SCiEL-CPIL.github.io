import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Build the target tensor constraints
    # T[i*m+j, j*p+k, k*n+i] = 1
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                constraints.append((u_idx, v_idx, w_idx))
    
    u_idxs = np.array([c[0] for c in constraints])
    v_idxs = np.array([c[1] for c in constraints])
    w_idxs = np.array([c[2] for c in constraints])
    
    def compute_error(U, V, W):
        """Compute sum of squared errors for all constraints."""
        vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
        return np.sum((vals - 1.0) ** 2)
    
    def verify_exact(U, V, W, tol=1e-10):
        """Check if the decomposition is exact (within tolerance)."""
        return compute_error(U, V, W) < tol
    
    def round_to_halfint(X):
        """Round to nearest half-integer."""
        return np.round(X * 2) / 2
    
    # Schoolbook rank-24 solution
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
    
    best_U, best_V, best_W = U_seed.copy(), V_seed.copy(), W_seed.copy()
    best_rank = rank_seed
    
    def try_rank_with_grad(r, num_restarts=20, max_steps=500, seed_base=0):
        """Try to find a rank-r decomposition using gradient descent."""
        if time.time() > deadline - 10:
            return None
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                return None
            
            # Initialization strategy:
            # Start from schoolbook-like structure but with random perturbations
            # and reduced rank
            np.random.seed(seed_base + restart)
            
            # Initialize U, V, W with small random values
            scale = 0.3
            U = np.random.randn(n * m, r) * scale
            V = np.random.randn(m * p, r) * scale
            W = np.random.randn(p * n, r) * scale
            
            # Adaptive gradient descent with momentum
            lr = 0.02
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            prev_err = compute_error(U, V, W)
            stuck_count = 0
            
            for step in range(max_steps):
                if time.time() > deadline - 10:
                    return None
                
                # Compute current values and errors
                vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
                diffs = 2 * (vals - 1.0)
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                np.add.at(dU, u_idxs, diffs[:, None] * V[v_idxs, :] * W[w_idxs, :])
                np.add.at(dV, v_idxs, diffs[:, None] * U[u_idxs, :] * W[w_idxs, :])
                np.add.at(dW, w_idxs, diffs[:, None] * U[u_idxs, :] * V[v_idxs, :])
                
                # Update with momentum
                vU = momentum * vU - lr * dU
                vV = momentum * vV - lr * dV
                vW = momentum * vW - lr * dW
                
                U += vU
                V += vV
                W += vW
                
                err = compute_error(U, V, W)
                
                # Check for convergence
                if err < 1e-12:
                    # Round to nearest half-integer
                    U_r = round_to_halfint(U)
                    V_r = round_to_halfint(V)
                    W_r = round_to_halfint(W)
                    
                    if verify_exact(U_r, V_r, W_r):
                        return U_r, V_r, W_r
                    else:
                        # Try tighter rounding or different approach
                        break
                
                # Adaptive learning rate
                if err > prev_err:
                    lr *= 0.9
                    stuck_count += 1
                    if stuck_count > 50:
                        break
                else:
                    stuck_count = 0
                    if step > 100:
                        lr *= 1.01  # Slowly increase if making progress
                
                prev_err = err
        
        return None
    
    # Try ranks from 20 to 23
    for r in [20, 21, 22, 23]:
        if time.time() > deadline - 20:
            break
        
        sol = try_rank_with_grad(r, num_restarts=30, max_steps=400, seed_base=r * 1000)
        if sol is not None:
            U_r, V_r, W_r = sol
            # Verify once more
            if verify_exact(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = r
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
