"""Matrix multiplication 3,3,6 rank-minimal exact decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 6
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def verify(U, V, W):
        """Check exact decomposition."""
        rank = len(U[0]) if U else 0
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if not U or not V or not W:
            return False
        if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
            return False
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True

    def snap_and_verify(U, V, W):
        """Snap to half-integers or integers and verify."""
        for scale in [2, 1]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            U_list = U_r.tolist()
            V_list = V_r.tolist()
            W_list = W_r.tolist()
            if verify(U_list, V_list, W_list):
                return (U_list, V_list, W_list)
        return None

    # Try a structured approach: the 3x3x6 multiplication can be seen as
    # 3 separate 3x3x3 multiplications (one for each column block of the 6-column matrix)
    # that share the same A matrix.
    
    # Let's try to find a low-rank decomposition using the fact that
    # the tensor T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B
    # contributes to (i,k) entry of C.
    
    # The key insight: for 3x3x6, we can think of it as computing C = A*B where
    # B is 3x6. This is equivalent to 3 separate 3x3 multiplications where
    # A is shared but B has different column blocks.
    
    # Let's try rank 27 (which is 3 * 9, the schoolbook for 3 separate 3x3x3)
    # and see if we can do better.
    
    # Actually, let's try a smarter approach. The multiplication tensor for 3x3x6
    # has rank at most 54 (schoolbook). We want to find if it can be done with less.
    
    # Let's try random search with gradient descent for various ranks.
    
    def try_rank(r, time_budget):
        """Try to find rank-r decomposition via gradient descent."""
        end_time = start_time + time_budget
        idxs = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        best_err = float('inf')
        best_U = best_V = best_W = None

        # Number of restarts based on time budget
        num_restarts = max(1, int(time_budget / 0.1))
        num_restarts = min(num_restarts, 50)

        for restart in range(num_restarts):
            if time.time() > end_time:
                break
            
            # Random initialization
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # Add some structure to help convergence
            for t in range(r):
                i = t % n
                j = (t // n) % m
                k = (t // (n * m)) % p
                U[i * m + j, t] += 1.0
                V[j * p + k, t] += 1.0
                W[k * n + i, t] += 1.0

            for step in range(2000):
                if time.time() > end_time:
                    break
                
                # Compute residuals
                errs = np.zeros(n * m * p)
                for ii, (i, j, k) in enumerate(idxs):
                    errs[ii] = np.dot(U[i * m + j] * V[j * p + k], W[k * n + i]) - 1.0
                err = np.sum(errs ** 2)
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-12:
                    break
                
                # Gradient descent
                lr = 0.05 / (1 + step * 0.005)
                for ii, (i, j, k) in enumerate(idxs):
                    e = errs[ii]
                    if abs(e) > 1e-6:
                        U[i * m + j] -= lr * e * V[j * p + k] * W[k * n + i]
                        V[j * p + k] -= lr * e * U[i * m + j] * W[k * n + i]
                        W[k * n + i] -= lr * e * U[i * m + j] * V[j * p + k]

            if best_err < 1e-10:
                break

        if best_err < 1e-10:
            result = snap_and_verify(best_U, best_V, best_W)
            if result is not None:
                return result
        return None

    # Try ranks from lower bound upward
    # Lower bound considerations:
    # - The tensor has 54 non-zero entries
    # - Rank must be at least max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # - But the actual rank is likely higher
    
    # Try a few promising ranks
    ranks_to_try = [18, 20, 22, 24, 27, 30, 36, 45, 54]
    
    for r in ranks_to_try:
        if time.time() > deadline - 20:
            break
        time_per_rank = min(10.0, (deadline - time.time() - 15) / max(1, len(ranks_to_try) - ranks_to_try.index(r)))
        time_per_rank = max(time_per_rank, 2.0)
        
        result = try_rank(r, time_per_rank)
        if result is not None:
            U_list, V_list, W_list = result
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: schoolbook decomposition (rank 54)
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
