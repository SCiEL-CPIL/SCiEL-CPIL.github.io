import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety

    # Target tensor T[i, j, k] = 1
    # We want sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1
    
    best_rank = 17
    best_U, best_V, best_W = None, None, None
    
    # Store the current rank 17 solution as fallback
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))
    
    def set_U17(i, j, r, val): U17[i * m + j, r] = val
    def set_V17(j, k, r, val): V17[j * p + k, r] = val
    def set_W17(k, i, r, val): W17[k * n + i, r] = val

    # Copy rank 17 solution from the prompt
    set_U17(0, 0, 0, 1); set_U17(1, 1, 0, 1)
    set_V17(0, 0, 0, 1); set_V17(1, 1, 0, 1)
    set_W17(0, 0, 0, 1); set_W17(1, 1, 0, 1)
    
    set_U17(1, 0, 1, 1); set_U17(1, 1, 1, 1)
    set_V17(0, 0, 1, 1)
    set_W17(0, 1, 1, 1); set_W17(1, 1, 1, -1)
    
    set_U17(0, 0, 2, 1)
    set_V17(0, 1, 2, 1); set_V17(1, 1, 2, -1)
    set_W17(1, 0, 2, 1); set_W17(1, 1, 2, 1)
    
    set_U17(1, 1, 3, 1)
    set_V17(1, 0, 3, 1); set_V17(0, 0, 3, -1)
    set_W17(0, 0, 3, 1); set_W17(0, 1, 3, 1)
    
    set_U17(0, 0, 4, 1); set_U17(0, 1, 4, 1)
    set_V17(1, 1, 4, 1)
    set_W17(0, 0, 4, -1); set_W17(1, 0, 4, 1)
    
    set_U17(1, 0, 5, 1); set_U17(0, 0, 5, -1)
    set_V17(0, 0, 5, 1); set_V17(0, 1, 5, 1)
    set_W17(1, 1, 5, 1)
    
    set_U17(0, 1, 6, 1); set_U17(1, 1, 6, -1)
    set_V17(1, 0, 6, 1); set_V17(1, 1, 6, 1)
    set_W17(0, 0, 6, 1)
    
    set_U17(0, 0, 7, 1)
    set_V17(0, 2, 7, 1)
    set_W17(2, 0, 7, 1)
    
    set_U17(0, 1, 8, 1)
    set_V17(1, 2, 8, 1)
    set_W17(2, 0, 8, 1)
    
    set_U17(1, 0, 9, 1)
    set_V17(0, 2, 9, 1)
    set_W17(2, 1, 9, 1)
    
    set_U17(1, 1, 10, 1)
    set_V17(1, 2, 10, 1)
    set_W17(2, 1, 10, 1)
    
    set_U17(0, 2, 11, 1)
    set_V17(2, 0, 11, 1)
    set_W17(0, 0, 11, 1)
    
    set_U17(0, 2, 12, 1)
    set_V17(2, 1, 12, 1)
    set_W17(1, 0, 12, 1)
    
    set_U17(0, 2, 13, 1)
    set_V17(2, 2, 13, 1)
    set_W17(2, 0, 13, 1)
    
    set_U17(1, 2, 14, 1)
    set_V17(2, 0, 14, 1)
    set_W17(0, 1, 14, 1)
    
    set_U17(1, 2, 15, 1)
    set_V17(2, 1, 15, 1)
    set_W17(1, 1, 15, 1)
    
    set_U17(1, 2, 16, 1)
    set_V17(2, 2, 16, 1)
    set_W17(2, 1, 16, 1)
    
    best_U, best_V, best_W = U17, V17, W17

    def compute_error(U, V, W, rank):
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        
        # Vectorized error calculation
        # R[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # This is a bit tricky to fully vectorize without loops over i,j,k or r.
        # But we can do it in chunks.
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                # Sum over r and k
                # Term for fixed i, j:
                # s_k = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # We can compute this for all k at once?
                # V_j = V[j, :, :] shape (p, rank)
                # W_i = W[:, i, :] shape (p, rank) -> transpose to (rank, p)
                # U_i_j = U[i, j, :] shape (rank,)
                
                V_j = V_t[j, :, :] # (p, rank)
                W_i = W_t[:, i, :] # (p, rank)
                U_ij = U_t[i, j, :] # (rank,)
                
                # We want s_k = sum_r U_ij[r] * V_j[k, r] * W_i[k, r]
                # This is element-wise product of V_j and W_i, summed over r, weighted by U_ij.
                # P = V_j * W_i (element-wise) -> (p, rank)
                # s = P @ U_ij -> (p,)
                P = V_j * W_i
                s = P @ U_ij
                error += np.sum((s - 1.0) ** 2)
        return error

    def snap_to_half_integers(U, V, W):
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        return U_s, V_s, W_s

    def is_exact(U, V, W, rank):
        if compute_error(U, V, W, rank) < 1e-9:
            # Verify with exact integer arithmetic if possible?
            # For now, floating point check with tight tolerance is sufficient for the search.
            # The evaluator will check exactness.
            return True
        return False

    # Try to find lower rank
    for target_rank in [15, 16]:
        if time.time() > deadline:
            break
            
        for seed in range(20): # Reduced seeds to allow more steps per seed
            if time.time() > deadline:
                break
                
            np.random.seed(seed)
            
            # Initialize
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Local search / Gradient Descent
            lr = 0.05
            prev_error = float('inf')
            for step in range(500):
                if time.time() > deadline:
                    break
                    
                err = compute_error(U, V, W, target_rank)
                if err < 1e-10:
                    # Check if it snaps to half-integers
                    U_s, V_s, W_s = snap_to_half_integers(U, V, W)
                    if is_exact(U_s, V_s, W_s, target_rank):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_s, V_s, W_s
                        break
                    
                # Compute gradients
                U_t = U.reshape(n, m, target_rank)
                V_t = V.reshape(m, p, target_rank)
                W_t = W.reshape(p, n, target_rank)
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # s = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                            s = 0.0
                            for r in range(target_rank):
                                s += U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
                            res = s - 1.0
                            for r in range(target_rank):
                                u_val = U_t[i, j, r]
                                v_val = V_t[j, k, r]
                                w_val = W_t[k, i, r]
                                
                                dU[i * m + j, r] += 2 * res * v_val * w_val
                                dV[j * p + k, r] += 2 * res * u_val * w_val
                                dW[k * n + i, r] += 2 * res * u_val * v_val
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Snap occasionally
                if step % 20 == 0:
                    U, V, W = snap_to_half_integers(U, V, W)
                    if is_exact(U, V, W, target_rank):
                        best_rank = target_rank
                        best_U, best_V, best_W = U, V, W
                        break

            if best_rank == target_rank:
                break

    # Return best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
