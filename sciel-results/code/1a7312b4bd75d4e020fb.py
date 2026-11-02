"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    # Build the target tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Start with schoolbook as fallback
    U_fb = np.zeros((n * m, rank_schoolbook))
    V_fb = np.zeros((m * p, rank_schoolbook))
    W_fb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # Precompute index arrays for vectorized operations
    # For each (i,j,k), we need indices:
    #   u_idx = i*m + j
    #   v_idx = j*p + k  
    #   w_idx = k*n + i
    i_vals, j_vals, k_vals = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing='ij'
    )
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()
    u_idx = (i_flat * m + j_flat).astype(int)
    v_idx = (j_flat * p + k_flat).astype(int)
    w_idx = (k_flat * n + i_flat).astype(int)
    num_constraints = len(u_idx)  # n*m*p = 90
    
    def compute_error_fast(U, V, W):
        """Compute sum of squared errors efficiently using vectorization."""
        # For each constraint c (index into u_idx, v_idx, w_idx):
        #   s_c = sum_rank U[u_idx[c], :] * V[v_idx[c], :] * W[w_idx[c], :]
        # We want all s_c = 1
        # 
        # Efficient approach: for each rank component r, compute the contribution
        # across all constraints, then sum over r
        r = U.shape[1]
        # U_sel: (num_constraints, r) - U[u_idx[c], r] for each constraint c
        U_sel = U[u_idx]  # shape (num_constraints, r)
        V_sel = V[v_idx]  # shape (num_constraints, r)
        W_sel = W[w_idx]  # shape (num_constraints, r)
        # s_c = sum_r U_sel[c,r] * V_sel[c,r] * W_sel[c,r]
        s = np.sum(U_sel * V_sel * W_sel, axis=1)  # shape (num_constraints,)
        err = np.sum((s - 1.0) ** 2)
        return err
    
    def compute_gradients(U, V, W):
        """Compute gradients of the error with respect to U, V, W using vectorization."""
        r = U.shape[1]
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        U_sel = U[u_idx]  # (num_constraints, r)
        V_sel = V[v_idx]  # (num_constraints, r)
        W_sel = W[w_idx]  # (num_constraints, r)
        
        s = np.sum(U_sel * V_sel * W_sel, axis=1)  # (num_constraints,)
        err_term = 2.0 * (s - 1.0)  # (num_constraints,)
        
        # dU[u_idx[c], r] += err_term[c] * V_sel[c,r] * W_sel[c,r]
        # Use np.add.at for scattered updates
        dU_contrib = err_term[:, np.newaxis] * V_sel * W_sel  # (num_constraints, r)
        np.add.at(dU, u_idx, dU_contrib)
        
        dV_contrib = err_term[:, np.newaxis] * U_sel * W_sel  # (num_constraints, r)
        np.add.at(dV, v_idx, dV_contrib)
        
        dW_contrib = err_term[:, np.newaxis] * U_sel * V_sel  # (num_constraints, r)
        np.add.at(dW, w_idx, dW_contrib)
        
        return dU, dV, dW
    
    def adam_step(params, grads, m_params, v_params, t, beta1, beta2, lr, eps=1e-8):
        """Perform an Adam update step. params and grads are lists of arrays."""
        m_params[:] = beta1 * m_params + (1 - beta1) * grads
        v_params[:] = beta2 * v_params + (1 - beta2) * (grads ** 2)
        m_hat = m_params / (1 - beta1 ** t)
        v_hat = v_params / (1 - beta2 ** t)
        for i in range(len(params)):
            params[i] -= lr * m_hat[i] / (np.sqrt(v_hat[i]) + eps)
        return m_params, v_params
    
    # Try ranks from just below schoolbook down to some minimum
    for r in range(min(rank_schoolbook - 1, 85), 50, -1):
        if time.time() > deadline - 30:
            break
        
        best_err_for_r = float('inf')
        best_U_r, best_V_r, best_W_r = None, None, None
        
        # Multiple random restarts with different initialization strategies
        n_restarts = 8
        for restart in range(n_restarts):
            if time.time() > deadline - 20:
                break
            
            # Different initialization strategies
            if restart == 0:
                # Small random
                U = np.random.randn(n * m, r) * 0.1
                V = np.random.randn(m * p, r) * 0.1
                W = np.random.randn(p * n, r) * 0.1
            elif restart == 1:
                # Larger random
                U = np.random.randn(n * m, r) * 0.5
                V = np.random.randn(m * p, r) * 0.5
                W = np.random.randn(p * n, r) * 0.5
            elif restart == 2:
                # Sparse random
                U = np.random.randn(n * m, r) * 0.1
                V = np.random.randn(m * p, r) * 0.1
                W = np.random.randn(p * n, r) * 0.1
                # Zero out some entries
                mask = np.random.rand(n * m, r) < 0.5
                U[mask] = 0
                mask = np.random.rand(m * p, r) < 0.5
                V[mask] = 0
                mask = np.random.rand(p * n, r) < 0.5
                W[mask] = 0
            elif restart == 3:
                # Binary-like initialization
                U = np.random.choice([-1, 1], size=(n * m, r)) * 0.1
                V = np.random.choice([-1, 1], size=(m * p, r)) * 0.1
                W = np.random.choice([-1, 1], size=(p * n, r)) * 0.1
            elif restart == 4:
                # Structured: try to initialize closer to schoolbook pattern
                U = np.zeros((n * m, r))
                V = np.zeros((m * p, r))
                W = np.zeros((p * n, r))
                # Assign each rank component to cover some (i,j,k) combinations
                for c in range(r):
                    i = c % n
                    j = (c // n) % m
                    k = (c // (n * m)) % p
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                # Add small random perturbation
                U += np.random.randn(n * m, r) * 0.05
                V += np.random.randn(m * p, r) * 0.05
                W += np.random.randn(p * n, r) * 0.05
            else:
                # Mix of strategies
                scale = 0.1 + 0.1 * restart
                U = np.random.randn(n * m, r) * scale
                V = np.random.randn(m * p, r) * scale
                W = np.random.randn(p * n, r) * scale
            
            # Adam optimizer
            beta1, beta2 = 0.9, 0.999
            lr = 0.01
            m_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_U = np.zeros_like(U)
            v_V = np.zeros_like(V)
            v_W = np.zeros_like(W)
            
            best_local_err = float('inf')
            best_local_U, best_local_V, best_local_W = U.copy(), V.copy(), W.copy()
            
            for step in range(300):
                if time.time() > deadline - 10:
                    break
                
                dU, dV, dW = compute_gradients(U, V, W)
                
                params = [U, V, W]
                grads = [dU, dV, dW]
                m_params = [m_U, m_V, m_W]
                v_params = [v_U, v_V, v_W]
                
                m_U, m_V, m_W = m_params
                v_U, v_V, v_W = v_params
                
                # Adam update
                m_U = beta1 * m_U + (1 - beta1) * dU
                m_V = beta1 * m_V + (1 - beta1) * dV
                m_W = beta1 * m_W + (1 - beta1) * dW
                v_U = beta2 * v_U + (1 - beta2) * (dU ** 2)
                v_V = beta2 * v_V + (1 - beta2) * (dV ** 2)
                v_W = beta2 * v_W + (1 - beta2) * (dW ** 2)
                
                t = step + 1
                m_hat_U = m_U / (1 - beta1 ** t)
                m_hat_V = m_V / (1 - beta1 ** t)
                m_hat_W = m_W / (1 - beta1 ** t)
                v_hat_U = v_U / (1 - beta2 ** t)
                v_hat_V = v_V / (1 - beta2 ** t)
                v_hat_W = v_W / (1 - beta2 ** t)
                
                U -= lr * m_hat_U / (np.sqrt(v_hat_U) + 1e-8)
                V -= lr * m_hat_V / (np.sqrt(v_hat_V) + 1e-8)
                W -= lr * m_hat_W / (np.sqrt(v_hat_W) + 1e-8)
                
                # Check error periodically
                if step % 30 == 29:
                    err = compute_error_fast(U, V, W)
                    if err < best_local_err:
                        best_local_err = err
                        best_local_U = U.copy()
                        best_local_V = V.copy()
                        best_local_W = W.copy()
                    if err < 1e-6:
                        break
                    # Reduce learning rate if error is not decreasing
                    lr *= 0.85
            
            err = compute_error_fast(U, V, W)
            if err < best_local_err:
                best_local_err = err
                best_local_U = U.copy()
                best_local_V = V.copy()
                best_local_W = W.copy()
            
            if best_local_err < best_err_for_r:
                best_err_for_r = best_local_err
                best_U_r = best_local_U
                best_V_r = best_local_V
                best_W_r = best_local_W
        
        if best_err_for_r < 1e-6 and best_U_r is not None:
            best_rank = r
            best_U, best_V, best_W = best_U_r, best_V_r, best_W_r
            # Check if we can go lower
            continue
        else:
            # If we couldn't achieve low error at this rank, stop
            if best_err_for_r > 1e-3:
                break
    
    # Convert to lists of lists with integer/half-integer entries if possible
    # Round to nearest half-integer
    def to_half_int(arr):
        """Round to nearest half-integer."""
        return np.round(arr * 2) / 2.0
    
    # Check if the rounded version still works
    U_rounded = to_half_int(best_U)
    V_rounded = to_half_int(best_V)
    W_rounded = to_half_int(best_W)
    
    err_rounded = compute_error_fast(U_rounded, V_rounded, W_rounded)
    if err_rounded < 1e-9:
        U_final, V_final, W_final = U_rounded, V_rounded, W_rounded
    else:
        # Use the unrounded version if rounding breaks it
        # But we need integer/half-integer entries...
        # Try to find a better solution or use schoolbook
        if best_rank < rank_schoolbook:
            # Try harder to get exact half-integer solution
            # For now, fall back to schoolbook if we can't get exact
            U_final, V_final, W_final = U_fb, V_fb, W_fb
        else:
            U_final, V_final, W_final = U_fb, V_fb, W_fb
    
    # Convert numpy arrays to lists of lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    # Ensure all values are integers or half-integers
    for mat in [U_list, V_list, W_list]:
        for row in mat:
            for idx, val in enumerate(row):
                # Round to nearest half-integer
                row[idx] = round(val * 2) / 2.0
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
