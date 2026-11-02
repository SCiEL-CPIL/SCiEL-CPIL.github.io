"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]
DEADLINE = 145.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Strategy: direct TTD-style continuous optimization for 3x3x7
    # Start with schoolbook rank 63, try to reduce rank
    # Use the fact that the multiplication tensor has structure
    
    def schoolbook_decomp(n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    def verify_decomp(U, V, W, n, m, p, tol=1e-9):
        """Verify that U, V, W form an exact decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    if abs(val - 1) > tol:
                        return False
        return True
    
    def compute_error(U, V, W, n, m, p):
        """Compute the max error of the decomposition."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err = abs(val - 1)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def polish(U, V, W, n, m, p, max_iter=5000, tol=1e-12):
        """Polish a near-exact decomposition to exact using gradient descent."""
        U = U.copy().astype(np.float64)
        V = V.copy().astype(np.float64)
        W = W.copy().astype(np.float64)
        rank = U.shape[1]
        
        for it in range(max_iter):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Compute gradients for all (i,j,k)
            U_grad = np.zeros_like(U)
            V_grad = np.zeros_like(V)
            W_grad = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_row = U[i*m + j, :]
                        v_row = V[j*p + k, :]
                        w_row = W[k*n + i, :]
                        
                        val = np.dot(u_row, v_row * w_row)
                        err = 1.0 - val
                        
                        # d/dU = v_row * w_row * err
                        # d/dV = u_row * w_row * err
                        # d/dW = u_row * v_row * err
                        U_grad[i*m + j, :] += v_row * w_row * err
                        V_grad[j*p + k, :] += u_row * w_row * err
                        W_grad[k*n + i, :] += u_row * v_row * err
            
            max_err = compute_error(U, V, W, n, m, p)
            if max_err < tol:
                return U, V, W, True
            
            # Adaptive learning rate
            max_grad = max(np.max(np.abs(U_grad)), np.max(np.abs(V_grad)), np.max(np.abs(W_grad)))
            if max_grad < 1e-15:
                break
            
            lr = 0.01 * (1.0 + it / max_iter)
            U -= lr * U_grad
            V -= lr * V_grad
            W -= lr * W_grad
        
        final_err = compute_error(U, V, W, n, m, p)
        return U, V, W, final_err < tol
    
    # Get schoolbook decomposition as starting point
    U_sb, V_sb, W_sb, rank_sb = schoolbook_decomp(n, m, p)
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Try to reduce rank by starting from schoolbook and zeroing random columns
    # then polishing
    np.random.seed(42)
    
    for target_rank in range(best_rank - 1, 30, -1):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        attempts = 0
        max_attempts = 50
        success = False
        
        for attempt in range(max_attempts):
            if time.time() - start_time > DEADLINE - 20:
                break
            
            # Start from schoolbook, zero out (rank_sb - target_rank) random columns
            U_try = U_sb.copy()
            V_try = V_sb.copy()
            W_try = W_sb.copy()
            
            # Randomly zero columns
            cols_to_zero = np.random.choice(rank_sb, size=rank_sb - target_rank, replace=False)
            U_try[:, cols_to_zero] = 0
            V_try[:, cols_to_zero] = 0
            W_try[:, cols_to_zero] = 0
            
            # Add small random perturbation to remaining columns
            mask = np.ones(rank_sb, dtype=bool)
            mask[cols_to_zero] = False
            U_try[:, mask] += np.random.randn(n*m, target_rank) * 0.1
            V_try[:, mask] += np.random.randn(m*p, target_rank) * 0.1
            W_try[:, mask] += np.random.randn(p*n, target_rank) * 0.1
            
            # Polish
            U_pol, V_pol, W_pol, valid = polish(U_try, V_try, W_try, n, m, p, max_iter=2000)
            
            if valid:
                best_rank = target_rank
                best_U, best_V, best_W = U_pol, V_pol, W_pol
                success = True
                break
        
        if not success:
            # Try a different strategy: random initialization
            for attempt in range(10):
                if time.time() - start_time > DEADLINE - 20:
                    break
                
                U_try = np.random.randn(n*m, target_rank) * 0.5
                V_try = np.random.randn(m*p, target_rank) * 0.5
                W_try = np.random.randn(p*n, target_rank) * 0.5
                
                U_pol, V_pol, W_pol, valid = polish(U_try, V_try, W_try, n, m, p, max_iter=3000)
                
                if valid:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_pol, V_pol, W_pol
                    success = True
                    break
            
            if not success:
                # If we can't reduce further, stop
                if target_rank < best_rank - 2:
                    break
    
    # Final verification
    if not verify_decomp(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook
        best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
        best_rank = rank_sb
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

