"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    # Set deadline to 100s to ensure we return before the 120s kill
    deadline = start_time + 100.0

    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = make_schoolbook()
    best_rank = n * m * p

    # Precompute index arrays for efficient verification
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx

    def verify(U, V, W, r):
        """Check if U, V, W form a valid tensor decomposition. Returns max error."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        u_vals = Ua[u_idx, :]  # (n*m*p, r)
        v_vals = Va[v_idx, :]  # (n*m*p, r)
        w_vals = Wa[w_idx, :]  # (n*m*p, r)
        
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        
        return np.max(np.abs(result - 1.0))

    def refine_vectorized(U, V, W, r, max_iter=100):
        """
        Refine using vectorized gradient descent.
        """
        Ua = np.array(U, dtype=float).copy()
        Va = np.array(V, dtype=float).copy()
        Wa = np.array(W, dtype=float).copy()
        
        # Momentum buffers
        mU = np.zeros_like(Ua)
        mV = np.zeros_like(Va)
        mW = np.zeros_like(Wa)
        beta = 0.9
        lr = 0.05
        
        for it in range(max_iter):
            if time.time() > deadline - 2:
                break
            
            # Compute current values
            u_vals = Ua[u_idx, :]  # (N, r)
            v_vals = Va[v_idx, :]  # (N, r)
            w_vals = Wa[w_idx, :]  # (N, r)
            
            # Reshape to 4D for easier gradient calculation
            # N = n*m*p
            u_4d = u_vals.reshape(n, m, p, r)
            v_4d = v_vals.reshape(n, m, p, r)
            w_4d = w_vals.reshape(n, m, p, r)
            
            prod = u_4d * v_4d * w_4d
            result = prod.sum(axis=3) # (n, m, p)
            error = result - 1.0
            err_norm = np.linalg.norm(error)
            
            if err_norm < 1e-8:
                break
            
            # Gradients
            # L = sum_{i,j,k} (sum_c U[i,j,c] V[j,k,c] W[k,i,c] - 1)^2
            # dL/dU[i,j,c] = 2 * sum_k error[i,j,k] * V[j,k,c] * W[k,i,c]
            
            # Vectorized Gradient for U
            # We need sum over k.
            # err_4d is (n, m, p, 1)
            err_4d = error[:, :, :, np.newaxis]
            
            # grad_U[i,j,c] = sum_k 2 * err[i,j,k] * v[j,k,c] * w[k,i,c]
            # v_4d is (n, m, p, r). We need to sum over p (which is k).
            # But w_4d is (n, m, p, r) where w is indexed by (k, i).
            # In w_4d, the axis 0 is i, axis 1 is j, axis 2 is k.
            # Wait, w_vals is Wa[w_idx, :]. w_idx is k*n + i.
            # So w_4d[i, j, k, c] corresponds to W[k*n+i, c].
            
            # grad_U[i, j, c] = sum_k 2 * error[i, j, k] * v_4d[i, j, k, c] * w_4d[i, j, k, c]
            # This is simply: 2 * sum_k (error * v * w) along axis 2
            grad_U_4d = 2 * err_4d * v_4d * w_4d
            grad_U = np.sum(grad_U_4d, axis=2).reshape(n * m, r)
            
            # grad_V[j, k, c] = sum_i 2 * error[i, j, k] * u_4d[i, j, k, c] * w_4d[i, j, k, c]
            # v_4d[i, j, k, c] corresponds to V[j*p+k, c].
            # We need to sum over i (axis 0).
            # The result should be indexed by (j, k).
            # So we sum axis 0 of (2 * err * u * w).
            grad_V_4d = 2 * err_4d * u_4d * w_4d
            # grad_V_4d shape is (n, m, p, r).
            # We want to sum over i (axis 0).
            # Result shape (m, p, r).
            grad_V_sum = np.sum(grad_V_4d, axis=0)
            grad_V = grad_V_sum.reshape(m * p, r)
            
            # grad_W[k, i, c] = sum_j 2 * error[i, j, k] * u_4d[i, j, k, c] * v_4d[i, j, k, c]
            # w_4d[i, j, k, c] corresponds to W[k*n+i, c].
            # We need to sum over j (axis 1).
            # Result shape (n, p, r) -> but indexed by (k, i).
            # So we need to transpose or reorder.
            # grad_W_4d shape (n, m, p, r).
            # Sum over axis 1 (j). Result (n, p, r).
            # We need to map (n, p) to (p*n) index k*n+i.
            # Current axes: 0 is i, 1 is p (k).
            # We want output index k*n + i.
            # So we need to transpose to (p, n) then reshape.
            grad_W_4d = 2 * err_4d * u_4d * v_4d
            grad_W_sum = np.sum(grad_W_4d, axis=1) # Shape (n, p, r)
            # Transpose to (p, n, r)
            grad_W_sum = np.transpose(grad_W_sum, (1, 0, 2))
            grad_W = grad_W_sum.reshape(p * n, r)
            
            # Update with momentum
            mU = beta * mU + (1 - beta) * grad_U
            mV = beta * mV + (1 - beta) * grad_V
            mW = beta * mW + (1 - beta) * grad_W
            
            Ua -= lr * mU
            Va -= lr * mV
            Wa -= lr * mW
            
            # Decay learning rate
            lr *= 0.99
        
        return Ua, Va, Wa

    # Try to find lower rank decompositions
    # We will try a few ranks slightly below schoolbook to see if we can find a solution.
    # However, random search is unlikely to find a significantly lower rank for this problem.
    # We will focus on speed and reliability.
    
    # If we have time, try rank = n*m*p - 1, etc? 
    # Actually, let's just try a few random initializations for a few lower ranks.
    
    candidate_ranks = [best_rank - 1, best_rank - 2, best_rank - 5]
    
    for r in candidate_ranks:
        if time.time() > deadline - 5:
            break
        
        for trial in range(5):
            if time.time() > deadline - 2:
                break
            
            np.random.seed(trial * 1000 + r)
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            U_ref, V_ref, W_ref = refine_vectorized(U, V, W, r, max_iter=50)
            
            err = verify(U_ref, V_ref, W_ref, r)
            if err < 1e-6:
                # Check if we can round to exact integers/halves
                # Simple rounding check
                U_round = np.round(U_ref * 2) / 2
                V_round = np.round(V_ref * 2) / 2
                W_round = np.round(W_ref * 2) / 2
                err_round = verify(U_round, V_round, W_round, r)
                if err_round < 1e-9:
                    best_rank = r
                    best_result = {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
                    break
        
        if best_rank < n * m * p:
            break
            
    return best_result
# EVOLVE-BLOCK-END
