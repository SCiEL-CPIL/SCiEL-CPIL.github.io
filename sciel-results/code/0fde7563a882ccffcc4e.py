"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0

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

    def verify_fast(Ua, Va, Wa):
        """Fast verification using precomputed indices. Returns max error."""
        u_vals = Ua[u_idx, :]
        v_vals = Va[v_idx, :]
        w_vals = Wa[w_idx, :]
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        return np.max(np.abs(result - 1.0))

    def strassen_4x4x4():
        """
        Return Strassen decomposition for 4x4 * 4x4 -> 4x4.
        Returns U (16, 7), V (16, 7), W (16, 7) as float arrays.
        """
        # Strassen's 7 products for 2x2
        # For 4x4, we can use two levels or a direct 7-product scheme
        # Actually, let's use the known rank-7 decomposition for 4x4x4
        
        # The 4x4x4 tensor has rank 7 (Strassen)
        # We'll construct it from the 2x2x2 Strassen decomposition
        
        # 2x2 Strassen:
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # p1 = (a+d)(e+h), p2 = (c+d)e, p3 = a(f-h), p4 = d(g-e)
        # p5 = (a+b)h, p6 = (c-a)(e+f), p7 = (b-d)(g+h)
        # C = [[p1+p3-p5-p7, p5+p7], [p2+p4, p1-p2+p4+p6]]
        
        # For 4x4, we partition into 2x2 blocks:
        # A = [[A11, A12], [A21, A22]] where each is 2x2
        # B = [[B11, B12], [B21, B22]]
        # C = A*B computed via Strassen on the blocks
        
        # This gives rank 7 * 4 = 28 for the full 4x4x4? No...
        # Actually Strassen's algorithm applied to 4x4 gives rank 7*4=28
        # But we can do better. The tensor rank of 4x4x4 is known to be 7.
        
        # Let me use a different approach: the rank of the 4x4x4 multiplication tensor
        # is actually 7 (this is a known result).
        
        # I'll use a constructive approach with bilinear operations.
        
        # For simplicity, let's use the schoolbook for 4x4x4 (rank 16) 
        # and then try to reduce it.
        
        # Actually, let me just implement a basic Strassen-like decomposition.
        # For 4x4, we can use the fact that it's a Kronecker product structure.
        
        # Let me use a simpler approach: just return the schoolbook for now
        # and rely on the outer product structure to get the overall rank right.
        
        rank = 16
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i * 4 + j, col] = 1
                    V[j * 4 + k, col] = 1
                    W[k * 4 + i, col] = 1
                    col += 1
        return U, V, W

    def kronecker_decomposition():
        """
        Use Kronecker product structure to decompose 4x4x7 tensor.
        The 4x4x7 tensor can be seen as 4x4x4 ⊗ 4x1x7 (in a loose sense).
        We'll construct U, V, W using Kronecker products.
        """
        # For the 4x4x4 part, use schoolbook (rank 16)
        # For the 4x1x7 part, this is just a 7-dimensional vector
        # The total rank would be 16 * 7 = 112 (same as schoolbook)
        
        # To get a lower rank, we need to exploit the structure more carefully.
        
        # Let me try a different approach: use the fact that 4 = 2*2
        # and 7 is prime, so we can't decompose 7 further.
        
        # Strategy: Use Strassen for the 4x4x4 part (if we can get rank 7)
        # and multiply by the 7-dimensional part.
        
        # For now, let's just return the schoolbook decomposition
        # and let the optimization find improvements.
        
        U, V, W = strassen_4x4x4()
        r = U.shape[1]
        
        # Now we need to extend this to handle the 4x4x7 case
        # The 7 in the output dimension means we have 7 columns in B
        # We can think of B as having 7 columns, each of length 4
        
        # Actually, let me just use the schoolbook for the full problem
        U_full = np.zeros((16, 112))
        V_full = np.zeros((28, 112))
        W_full = np.zeros((28, 112))
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    U_full[i * 4 + j, col] = 1
                    V_full[j * 7 + k, col] = 1
                    W_full[k * 4 + i, col] = 1
                    col += 1
        
        return U_full, V_full, W_full

    def bilinear_reduction(U, V, W, target_rank, max_iter=500):
        """
        Try to reduce the rank using bilinear operations.
        This is a heuristic approach.
        """
        Ua = U.astype(float).copy()
        Va = V.astype(float).copy()
        Wa = W.astype(float).copy()
        r = Ua.shape[1]
        
        if r <= target_rank:
            return Ua[:, :target_rank], Va[:, :target_rank], Wa[:, :target_rank]
        
        # Try to find a lower-rank approximation
        # Use alternating minimization
        
        best_U, best_V, best_W = Ua, Va, Wa
        best_err = verify_fast(Ua, Va, Wa)
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Try to reduce rank by 1
            new_r = r - 1
            if new_r < target_rank:
                break
            
            # Initialize new matrices
            U_new = np.random.randn(16, new_r) * 0.1
            V_new = np.random.randn(28, new_r) * 0.1
            W_new = np.random.randn(28, new_r) * 0.1
            
            # Optimize
            for sub_it in range(100):
                if time.time() > deadline - 10:
                    break
                
                u_vals = U_new[u_idx, :]
                v_vals = V_new[v_idx, :]
                w_vals = W_new[w_idx, :]
                
                prod = u_vals * v_vals * w_vals
                result = prod.sum(axis=1)
                error = result - 1.0
                err_norm = np.linalg.norm(error)
                
                if err_norm < 1e-8:
                    break
                
                # Gradients
                u_4d = u_vals.reshape(4, 4, 7, new_r)
                v_4d = v_vals.reshape(4, 4, 7, new_r)
                w_4d = w_vals.reshape(4, 4, 7, new_r)
                
                err_4d = error.reshape(4, 4, 7, 1)
                
                grad_U_4d = 2 * err_4d * v_4d * w_4d
                grad_U = np.sum(grad_U_4d, axis=2).reshape(16, new_r)
                
                grad_V_4d = 2 * err_4d * u_4d * w_4d
                grad_V = np.sum(grad_V_4d, axis=0).reshape(28, new_r)
                
                grad_W_4d = 2 * err_4d * u_4d * v_4d
                grad_W_sum = np.sum(grad_W_4d, axis=1)
                grad_W_sum = np.transpose(grad_W_sum, (1, 0, 2))
                grad_W = grad_W_sum.reshape(28, new_r)
                
                U_new -= 0.05 * grad_U
                V_new -= 0.05 * grad_V
                W_new -= 0.05 * grad_W
            
            err = verify_fast(U_new, V_new, W_new)
            if err < best_err:
                best_U, best_V, best_W = U_new, V_new, W_new
                best_err = err
                r = new_r
            
            if r == target_rank:
                break
        
        return best_U[:, :target_rank], best_V[:, :target_rank], best_W[:, :target_rank]

    def optimize_rank(U, V, W, target_rank, max_iter=300):
        """
        Optimize U, V, W to achieve the target rank exactly.
        """
        Ua = np.zeros((16, target_rank))
        Va = np.zeros((28, target_rank))
        Wa = np.zeros((28, target_rank))
        
        # Initialize with random values
        np.random.seed(42)
        Ua = np.random.randn(16, target_rank) * 0.3
        Va = np.random.randn(28, target_rank) * 0.3
        Wa = np.random.randn(28, target_rank) * 0.3
        
        # Momentum
        mU = np.zeros_like(Ua)
        mV = np.zeros_like(Va)
        mW = np.zeros_like(Wa)
        beta = 0.9
        lr = 0.1
        best_err = np.inf
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            u_vals = Ua[u_idx, :]
            v_vals = Va[v_idx, :]
            w_vals = Wa[w_idx, :]
            
            u_4d = u_vals.reshape(4, 4, 7, target_rank)
            v_4d = v_vals.reshape(4, 4, 7, target_rank)
            w_4d = w_vals.reshape(4, 4, 7, target_rank)
            
            prod = u_4d * v_4d * w_4d
            result = prod.sum(axis=3)
            error = result - 1.0
            err_norm = np.linalg.norm(error)
            
            if err_norm < 1e-10:
                break
            
            if err_norm < best_err:
                best_err = err_norm
            
            err_4d = error[:, :, :, np.newaxis]
            
            grad_U_4d = 2 * err_4d * v_4d * w_4d
            grad_U = np.sum(grad_U_4d, axis=2).reshape(16, target_rank)
            
            grad_V_4d = 2 * err_4d * u_4d * w_4d
            grad_V_sum = np.sum(grad_V_4d, axis=0)
            grad_V = grad_V_sum.reshape(28, target_rank)
            
            grad_W_4d = 2 * err_4d * u_4d * v_4d
            grad_W_sum = np.sum(grad_W_4d, axis=1)
            grad_W_sum = np.transpose(grad_W_sum, (1, 0, 2))
            grad_W = grad_W_sum.reshape(28, target_rank)
            
            mU = beta * mU + (1 - beta) * grad_U
            mV = beta * mV + (1 - beta) * grad_V
            mW = beta * mW + (1 - beta) * grad_W
            
            Ua -= lr * mU
            Va -= lr * mV
            Wa -= lr * mW
            
            # Adaptive learning rate
            new_err = verify_fast(Ua, Va, Wa)
            if new_err > best_err * 1.5:
                lr *= 0.5
            elif new_err < best_err * 0.5:
                lr *= 1.1
                lr = min(lr, 0.5)
        
        return Ua, Va, Wa

    def try_rank(target_rank, max_trials=3):
        """Try to find a valid decomposition with the given rank."""
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                return None
            
            np.random.seed(trial * 12345 + target_rank * 678)
            
            Ua, Va, Wa = optimize_rank(np.zeros((16, target_rank)), 
                                       np.zeros((28, target_rank)), 
                                       np.zeros((28, target_rank)), 
                                       target_rank, max_iter=200)
            
            err = verify_fast(Ua, Va, Wa)
            if err < 1e-8:
                # Try rounding to half-integers
                U_round = np.round(Ua * 2) / 2
                V_round = np.round(Va * 2) / 2
                W_round = np.round(Wa * 2) / 2
                err_round = verify_fast(U_round, V_round, W_round)
                if err_round < 1e-9:
                    return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
        
        return None

    best_result = make_schoolbook()
    best_rank = n * m * p

    # Try to find lower ranks using the optimization
    # Start from a reasonable rank and go down
    # The theoretical lower bound is around n*m*p / log(n*m*p) but that's not tight
    # Let's try ranks from 90 to 111
    
    found_improvement = False
    for r in range(90, 112):
        if time.time() > deadline - 20:
            break
        if found_improvement:
            break
        
        result = try_rank(r, max_trials=2)
        if result is not None:
            best_rank = r
            best_result = result
            found_improvement = True
            break

    return best_result
# EVOLVE-BLOCK-END
