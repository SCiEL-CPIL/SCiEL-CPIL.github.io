"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Build the target tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i)
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
                
    T_target = np.zeros((n * m, m * p, p * n))
    for a, b, c in target_indices:
        T_target[a, b, c] = 1.0

    # Helper to check and snap
    def check_and_snap(U, V, W, r):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Snap to nearest half-integer
        U_snap = np.round(U_np * 2) / 2
        V_snap = np.round(V_np * 2) / 2
        W_snap = np.round(W_np * 2) / 2
        
        # Reconstruct
        # T_recon[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
        
        # Check exactness
        if not np.allclose(T_recon, T_target, atol=1e-9):
            return None
            
        return {
            "U": U_snap.tolist(),
            "V": V_snap.tolist(),
            "W": W_snap.tolist()
        }

    # Fast custom Gradient Descent for Tensor Decomposition
    def fast_gradient_descent(U, V, W, r, max_iters=5000, tol=1e-10):
        """
        Minimizes || T_target - sum_r U[:,r] V[:,r]^T W[:,r]^T ||_F^2
        U: (n*m, r), V: (m*p, r), W: (p*n, r)
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        lr = 0.01
        best_loss = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Compute Reconstruction
            # T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T_recon - T_target
            loss = np.sum(diff ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if loss < tol:
                break
                
            # Compute Gradients
            # dL/dU[a,r] = sum_{b,c} 2 * diff[a,b,c] * V[b,r] * W[c,r]
            # dL/dV[b,r] = sum_{a,c} 2 * diff[a,b,c] * U[a,r] * W[c,r]
            # dL/dW[c,r] = sum_{a,b} 2 * diff[a,b,c] * U[a,r] * V[b,r]
            
            # Efficient gradient calculation using einsum
            # grad_U[a,r] = 2 * sum_{b,c} diff[a,b,c] V[b,r] W[c,r]
            # We can compute this as 2 * einsum('abc,br,cr->ar', diff, V, W)
            grad_U = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
            grad_V = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
            grad_W = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Adaptive learning rate: if loss increases, reduce lr
            # (Simple heuristic: if we haven't improved in a while, reduce lr)
            # For simplicity and speed, we keep fixed lr but might reduce it if loss explodes
            if not np.isfinite(loss):
                break
                
            # Periodic lr reduction if stuck
            if it > 100 and it % 100 == 0 and loss > best_loss * 0.99:
                lr *= 0.95

        return best_U, best_V, best_W, best_loss

    # Fallback Schoolbook Rank 50 Solution
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1.0
                V_school[b, col] = 1.0
                W_school[c, col] = 1.0
                col += 1
                
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school

    # Strategy 1: Try to find Rank 49 by removing one term and optimizing.
    # We try removing each of the 50 terms.
    # To save time, we limit the iterations per attempt.
    
    for remove_idx in range(rank_school):
        if time.time() > deadline - 10:
            break
            
        # Create initial guess for Rank 49
        # We take all columns except 'remove_idx'
        keep_indices = [i for i in range(rank_school) if i != remove_idx]
        
        U_init = U_school[:, keep_indices]
        V_init = V_school[:, keep_indices]
        W_init = W_school[:, keep_indices]
        
        # Add a small random perturbation to help escape local minima
        # The removed term needs to be "absorbed" by the others.
        np.random.seed(remove_idx) # Deterministic but varied per attempt
        noise_scale = 0.1
        U_init = U_init + np.random.randn(*U_init.shape) * noise_scale
        V_init = V_init + np.random.randn(*V_init.shape) * noise_scale
        W_init = W_init + np.random.randn(*W_init.shape) * noise_scale
        
        # Run fast gradient descent
        # Limit iterations to ensure we finish in time
        # 50 attempts * (time per optimization)
        # If each takes ~1s, 50s total. Safe.
        U_sol, V_sol, W_sol, final_loss = fast_gradient_descent(U_init, V_init, W_init, 49, max_iters=2000)
        
        if final_loss < 1e-6:
            result = check_and_snap(U_sol, V_sol, W_sol, 49)
            if result:
                best_result = result
                best_rank = 49
                break
            # If snap fails, it might be close but not exact half-integers.
            # Try snapping with slightly looser tolerance? No, must be exact.
            # We can try to refine with a few more iterations if loss is very small
            elif final_loss < 1e-4:
                 # Try one more round with smaller lr? 
                 pass

    # Strategy 2: If Rank 49 failed, try Rank 48 with random initialization
    # This is less likely to succeed quickly, so we only do a few attempts.
    if best_rank == rank_school:
        for seed in [42, 123]:
            if time.time() > deadline - 5:
                break
            np.random.seed(seed)
            r = 48
            U_init = np.random.randn(n * m, r) * 0.5
            V_init = np.random.randn(m * p, r) * 0.5
            W_init = np.random.randn(p * n, r) * 0.5
            
            U_sol, V_sol, W_sol, final_loss = fast_gradient_descent(U_init, V_init, W_init, r, max_iters=1000)
            
            if final_loss < 1e-6:
                result = check_and_snap(U_sol, V_sol, W_sol, r)
                if result:
                    best_result = result
                    best_rank = r
                    break

    return best_result
# EVOLVE-BLOCK-END
