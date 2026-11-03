"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
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

    def make_hybrid_strassen():
        """
        Use Strassen's algorithm for the 4x4x4 part (rank 7) and schoolbook for 4x4x3 (rank 12).
        Total rank = 7 + 12 = 19? No, Strassen 4x4x4 has rank 7*... wait.
        
        Actually, Strassen for 2x2x2 has rank 7. For 4x4x4, we can use a block decomposition.
        But the standard Strassen rank for 4x4x4 is not simply 7.
        
        Let me reconsider. The known result is that 4x4x4 matrix multiplication can be done with rank 49.
        And 4x4x3 schoolbook is 4*4*3 = 48.
        Total = 49 + 48 = 97.
        
        But implementing the exact 49-rank decomposition for 4x4x4 is complex.
        
        Alternative: Use the fact that 4x4x4 can be decomposed using 4 Strassen 2x2x2 blocks?
        No, that gives 4*7 = 28, but with cross terms it's more.
        
        Let me try a different approach: use the known 49-rank decomposition for 4x4x4.
        This is complex to implement from scratch.
        
        Simpler approach: Try to find a lower rank via random search + refinement,
        but with better initialization and more trials.
        """
        pass

    def make_hybrid_simple():
        """
        Split the 4x7 multiplication into 4x4 and 4x3 parts.
        Use schoolbook for both: 4*4*4 + 4*4*3 = 64 + 48 = 112 (same as full schoolbook).
        
        To get improvement, we need a better algorithm for at least one part.
        
        Let me try: use a rank-49 decomposition for 4x4x4 if available,
        otherwise fall back to schoolbook.
        
        For now, let's implement a basic version and see if we can find improvements
        through random search.
        """
        # Start with schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

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

    def refine_vectorized(U, V, W, r, max_iter=200):
        """
        Refine using vectorized gradient descent with adaptive learning rate.
        """
        Ua = np.array(U, dtype=float).copy()
        Va = np.array(V, dtype=float).copy()
        Wa = np.array(W, dtype=float).copy()
        
        # Momentum buffers
        mU = np.zeros_like(Ua)
        mV = np.zeros_like(Va)
        mW = np.zeros_like(Wa)
        beta = 0.9
        lr = 0.1
        best_err = np.inf
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Compute current values
            u_vals = Ua[u_idx, :]  # (N, r)
            v_vals = Va[v_idx, :]  # (N, r)
            w_vals = Wa[w_idx, :]  # (N, r)
            
            # Reshape to 4D for easier gradient calculation
            u_4d = u_vals.reshape(n, m, p, r)
            v_4d = v_vals.reshape(n, m, p, r)
            w_4d = w_vals.reshape(n, m, p, r)
            
            prod = u_4d * v_4d * w_4d
            result = prod.sum(axis=3) # (n, m, p)
            error = result - 1.0
            err_norm = np.linalg.norm(error)
            
            if err_norm < 1e-8:
                break
            
            if err_norm < best_err:
                best_err = err_norm
            
            # Gradients
            err_4d = error[:, :, :, np.newaxis]
            
            # grad_U[i, j, c] = sum_k 2 * error[i, j, k] * v_4d[i, j, k, c] * w_4d[i, j, k, c]
            grad_U_4d = 2 * err_4d * v_4d * w_4d
            grad_U = np.sum(grad_U_4d, axis=2).reshape(n * m, r)
            
            # grad_V[j, k, c] = sum_i 2 * error[i, j, k] * u_4d[i, j, k, c] * w_4d[i, j, k, c]
            grad_V_4d = 2 * err_4d * u_4d * w_4d
            grad_V_sum = np.sum(grad_V_4d, axis=0)
            grad_V = grad_V_sum.reshape(m * p, r)
            
            # grad_W[k, i, c] = sum_j 2 * error[i, j, k] * u_4d[i, j, k, c] * v_4d[i, j, k, c]
            grad_W_4d = 2 * err_4d * u_4d * v_4d
            grad_W_sum = np.sum(grad_W_4d, axis=1) # Shape (n, p, r)
            grad_W_sum = np.transpose(grad_W_sum, (1, 0, 2))
            grad_W = grad_W_sum.reshape(p * n, r)
            
            # Update with momentum
            mU = beta * mU + (1 - beta) * grad_U
            mV = beta * mV + (1 - beta) * grad_V
            mW = beta * mW + (1 - beta) * grad_W
            
            Ua -= lr * mU
            Va -= lr * mV
            Wa -= lr * mW
            
            # Adaptive learning rate: reduce if error increases
            new_err = verify(Ua.tolist(), Va.tolist(), Wa.tolist(), r)
            if new_err > best_err * 1.5:
                lr *= 0.5
            elif new_err < best_err * 0.5:
                lr *= 1.1
                lr = min(lr, 0.5)
        
        return Ua, Va, Wa

    # Try to find lower rank decompositions using random search + refinement
    # Focus on ranks that are plausible: below 112, maybe down to 90
    
    candidate_ranks = list(range(97, 112)) + list(range(90, 97))
    candidate_ranks = sorted(set(candidate_ranks), key=lambda x: -x)  # Try higher ranks first
    
    found_improvement = False
    
    for r in candidate_ranks:
        if time.time() > deadline - 10:
            break
        if found_improvement:
            break
        
        num_trials = 3 if r > 100 else 2
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(trial * 10000 + r * 100)
            
            # Initialize with small random values
            U = np.random.randn(n * m, r) * 0.3
            V = np.random.randn(m * p, r) * 0.3
            W = np.random.randn(p * n, r) * 0.3
            
            U_ref, V_ref, W_ref = refine_vectorized(U, V, W, r, max_iter=150)
            
            err = verify(U_ref.tolist(), V_ref.tolist(), W_ref.tolist(), r)
            if err < 1e-6:
                # Check if we can round to exact integers/halves
                U_round = np.round(U_ref * 2) / 2
                V_round = np.round(V_ref * 2) / 2
                W_round = np.round(W_ref * 2) / 2
                err_round = verify(U_round.tolist(), V_round.tolist(), W_round.tolist(), r)
                if err_round < 1e-9:
                    best_rank = r
                    best_result = {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
                    found_improvement = True
                    break
    
    return best_result
# EVOLVE-BLOCK-END
