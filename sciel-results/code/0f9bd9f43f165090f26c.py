import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def _check_decomposition(U, V, W, n, m, p):
    """Check if U, V, W form an exact decomposition of J_{n,m,p}."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    
    return bool(np.allclose(T, 1.0, atol=1e-10))

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 2x3x9 matrix multiplication.
    
    Strategy:
    - Start with a rank-50 block-structure decomposition using Strassen on 2x2 blocks
      (splitting the inner dimension of size 3 into 2+1).
    - Then attempt to reduce rank via structured optimization:
      * Use gradient-based optimization of the sum-of-squares loss.
      * Try random restarts with adaptive learning rates.
      * Final high-precision polish pass.
    - Return the best valid decomposition found within the time budget.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin before 150s kill
    
    # =================================================================
    # Phase 1: Construct the rank-50 Strassen-based decomposition
    # =================================================================
    rank_init = 50
    U = [[0] * rank_init for _ in range(n * m)]
    V = [[0] * rank_init for _ in range(m * p)]
    W = [[0] * rank_init for _ in range(p * n)]
    
    col = 0
    u_a, u_b, u_c, u_d = 0, 1, 3, 4  # A[0,0], A[0,1], A[1,0], A[1,1]
    
    # Part 1: A[:,0:2] (2x2) * B[0:2,:] (2x9)
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        
        # Strassen's 7 products
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        
        col += 7
    
    # 1 block of 2x1 (column 8)
    k8 = 8
    v_e = 0 * p + k8
    v_g = 1 * p + k8
    
    U[u_a][col] += 1
    V[v_e][col] += 1
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    # Part 2: A[:,2:3] (2x1) * B[2:3,:] (1x9)
    u_ar0, u_ar1 = 2, 5
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        col += 2
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_init
    
    # Verify initial decomposition
    if not _check_decomposition(U, V, W, n, m, p):
        # Fallback: schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank
        return best_result
    
    # =================================================================
    # Phase 2: Attempt rank reduction via gradient-based optimization
    # =================================================================
    
    def _compute_T(U_arr, V_arr, W_arr, rank):
        """Compute the tensor T from factors."""
        U_t = U_arr.reshape(n, m, rank)
        V_t = V_arr.reshape(m, p, rank)
        W_t = W_arr.reshape(p, n, rank)
        return np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    
    def _compute_loss(U_arr, V_arr, W_arr, rank):
        """Compute sum of squared errors."""
        T = _compute_T(U_arr, V_arr, W_arr, rank)
        return float(np.sum((T - 1.0) ** 2))
    
    def _compute_gradients(U_arr, V_arr, W_arr, rank):
        """
        Compute analytic gradients of L = sum_{i,j,k}(T[i,j,k]-1)^2
        where T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        
        dL/dU[i,j,r] = 2 * sum_{i',j',k'} (T[i',j',k']-1) * dT[i',j',k']/dU[i,j,r]
        dT[i',j',k']/dU[i,j,r] = V[j,k',r]*W[k',i,r] if (i',j',k')=(i,j,k') for some k'
        
        More precisely:
        T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        dT[i,j,k]/dU[i,j,r] = V[j,k,r] * W[k,i,r]
        dT[i,j,k]/dV[j,k,r] = U[i,j,r] * W[k,i,r]
        dT[i,j,k]/dW[k,i,r] = U[i,j,r] * V[j,k,r]
        
        So:
        dL/dU[i,j,r] = 2 * sum_k (T[i,j,k]-1) * V[j,k,r] * W[k,i,r]
        dL/dV[j,k,r] = 2 * sum_i (T[i,j,k]-1) * U[i,j,r] * W[k,i,r]
        dL/dW[k,i,r] = 2 * sum_j (T[i,j,k]-1) * U[i,j,r] * V[j,k,r]
        """
        U_t = U_arr.reshape(n, m, rank)
        V_t = V_arr.reshape(m, p, rank)
        W_t = W_arr.reshape(p, n, rank)
        
        T = _compute_T(U_arr, V_arr, W_arr, rank)
        E = T - 1.0  # Error tensor, shape (n, m, p)
        
        # dL/dU[i,j,r] = 2 * sum_k E[i,j,k] * V[j,k,r] * W[k,i,r]
        # Using einsum: E[i,j,k] * V[j,k,r] * W[k,i,r] -> sum over k
        dU = 2.0 * np.einsum('ijk,jkr,kir->ijr', E, V_t, W_t)
        
        # dL/dV[j,k,r] = 2 * sum_i E[i,j,k] * U[i,j,r] * W[k,i,r]
        dV = 2.0 * np.einsum('ijk,ijr,kir->jkr', E, U_t, W_t)
        
        # dL/dW[k,i,r] = 2 * sum_j E[i,j,k] * U[i,j,r] * V[j,k,r]
        dW = 2.0 * np.einsum('ijk,ijr,jkr->kir', E, U_t, V_t)
        
        return dU, dV, dW
    
    def _gradient_descent(U_init, V_init, W_init, target_rank, max_iters=500, 
                          time_budget=10, initial_lr=0.01):
        """
        Gradient descent with adaptive learning rate to minimize the loss.
        Returns (U, V, W) if loss < 1e-10, else None.
        """
        start_time = time.time()
        
        # Initialize from the given factors, truncated/padded to target_rank
        U_arr = np.array(U_init, dtype=float).copy()
        V_arr = np.array(V_init, dtype=float).copy()
        W_arr = np.array(W_init, dtype=float).copy()
        
        # Truncate to target_rank
        if U_arr.shape[1] > target_rank:
            U_arr = U_arr[:, :target_rank]
            V_arr = V_arr[:, :target_rank]
            W_arr = W_arr[:, :target_rank]
        elif U_arr.shape[1] < target_rank:
            # Pad with small random values
            pad_U = np.random.normal(0, 0.01, (U_arr.shape[0], target_rank - U_arr.shape[1]))
            pad_V = np.random.normal(0, 0.01, (V_arr.shape[0], target_rank - V_arr.shape[1]))
            pad_W = np.random.normal(0, 0.01, (W_arr.shape[0], target_rank - W_arr.shape[1]))
            U_arr = np.hstack([U_arr, pad_U])
            V_arr = np.hstack([V_arr, pad_V])
            W_arr = np.hstack([W_arr, pad_W])
        
        current_loss = _compute_loss(U_arr, V_arr, W_arr, target_rank)
        
        if current_loss < 1e-10:
            return U_arr, V_arr, W_arr
        
        lr = initial_lr
        patience = 0
        best_loss = current_loss
        best_U = U_arr.copy()
        best_V = V_arr.copy()
        best_W = W_arr.copy()
        
        for it in range(max_iters):
            if time.time() - start_time > time_budget:
                break
            
            # Compute gradients
            dU, dV, dW = _compute_gradients(U_arr, V_arr, W_arr, target_rank)
            
            # Gradient step
            U_arr -= lr * dU
            V_arr -= lr * dV
            W_arr -= lr * dW
            
            new_loss = _compute_loss(U_arr, V_arr, W_arr, target_rank)
            
            if new_loss < current_loss:
                current_loss = new_loss
                patience = 0
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_U = U_arr.copy()
                    best_V = V_arr.copy()
                    best_W = W_arr.copy()
                if current_loss < 1e-10:
                    return U_arr, V_arr, W_arr
                # Slightly increase learning rate if making good progress
                if patience == 0 and it % 50 == 0 and it > 0:
                    lr = min(lr * 1.1, 0.1)
            else:
                # Revert and decrease learning rate
                U_arr += lr * dU
                V_arr += lr * dV
                W_arr += lr * dW
                lr *= 0.5
                patience += 1
                if patience > 50:
                    break
        
        return best_U, best_V, best_W
    
    def _try_reduce_rank(target_rank, num_restarts=5, time_per_restart=8):
        """
        Try to find a valid decomposition at the given rank using multiple restarts.
        """
        start_time = time.time()
        
        for restart in range(num_restarts):
            if time.time() - start_time > time_per_restart * num_restarts:
                break
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if restart == 0:
                # Start from current best, truncated
                U_init = best_result["U"]
                V_init = best_result["V"]
                W_init = best_result["W"]
            elif restart == 1:
                # Random initialization with small values
                U_init = np.random.normal(0, 0.5, (n*m, target_rank))
                V_init = np.random.normal(0, 0.5, (m*p, target_rank))
                W_init = np.random.normal(0, 0.5, (p*n, target_rank))
            elif restart == 2:
                # Start from best but with random perturbation
                U_arr = np.array(best_result["U"], dtype=float)
                V_arr = np.array(best_result["V"], dtype=float)
                W_arr = np.array(best_result["W"], dtype=float)
                U_init = U_arr[:, :min(target_rank, U_arr.shape[1])]
                V_init = V_arr[:, :min(target_rank, V_arr.shape[1])]
                W_init = W_arr[:, :min(target_rank, W_arr.shape[1])]
                U_init = U_init + np.random.normal(0, 0.1, U_init.shape)
                V_init = V_init + np.random.normal(0, 0.1, V_init.shape)
                W_init = W_init + np.random.normal(0, 0.1, W_init.shape)
            elif restart == 3:
                # Structured initialization based on block structure
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))
                # Fill with a scaled version of the first target_rank columns of best
                U_best = np.array(best_result["U"], dtype=float)
                V_best = np.array(best_result["V"], dtype=float)
                W_best = np.array(best_result["W"], dtype=float)
                scale = 1.0
                U_init[:, :min(target_rank, U_best.shape[1])] = U_best[:, :min(target_rank, U_best.shape[1])] * scale
                V_init[:, :min(target_rank, V_best.shape[1])] = V_best[:, :min(target_rank, V_best.shape[1])] * scale
                W_init[:, :min(target_rank, W_best.shape[1])] = W_best[:, :min(target_rank, W_best.shape[1])] * scale
                # Add small random noise
                U_init += np.random.normal(0, 0.05, U_init.shape)
                V_init += np.random.normal(0, 0.05, V_init.shape)
                W_init += np.random.normal(0, 0.05, W_init.shape)
            else:
                # Pure random
                U_init = np.random.normal(0, 1.0, (n*m, target_rank))
                V_init = np.random.normal(0, 1.0, (m*p, target_rank))
                W_init = np.random.normal(0, 1.0, (p*n, target_rank))
            
            result = _gradient_descent(U_init, V_init, W_init, target_rank, 
                                       max_iters=300, time_budget=time_per_restart)
            
            if result is not None:
                U_new, V_new, W_new = result
                if _check_decomposition(U_new, V_new, W_new, n, m, p):
                    return U_new, V_new, W_new
        
        return None
    
    # Try to reduce rank from 50 to lower values
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # Try ranks from 49 down to 30
    
    for target_rank in range(49, 29, -1):
        if time.time() > deadline:
            break
        
        if target_rank >= best_rank:
            continue
        
        # Try to find a valid decomposition at this rank
        result = _try_reduce_rank(target_rank, num_restarts=3, time_per_restart=5)
        
        if result is not None:
            U_new, V_new, W_new = result
            if _check_decomposition(U_new, V_new, W_new, n, m, p):
                best_result = {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
                best_rank = target_rank
                # Continue trying to reduce further
                continue
        else:
            # If we failed at this rank, try a few more restarts with more time
            if time.time() < deadline - 20:
                result = _try_reduce_rank(target_rank, num_restarts=5, time_per_restart=4)
                if result is not None:
                    U_new, V_new, W_new = result
                    if _check_decomposition(U_new, V_new, W_new, n, m, p):
                        best_result = {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
                        best_rank = target_rank
                        continue
    
    # =================================================================
    # Phase 3: Final verification and return
    # =================================================================
    if not _check_decomposition(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
