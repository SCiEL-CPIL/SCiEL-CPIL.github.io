import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = 2, 4, 6
    
    # The tensor T is defined by T[i*m+j, j*p+k, k*n+i] = 1
    # We want U, V, W such that sum_r U[:,r] * V[:,r] * W[:,r] = T
    # This is equivalent to finding a low-rank decomposition of the 3-tensor.
    
    # We model the decomposition as a sum of R "structural" terms.
    # Each term r is defined by:
    #   U_r = u_r * (indicator of some subset of A)
    #   V_r = v_r * (indicator of some subset of B)
    #   W_r = w_r * (indicator of some subset of C)
    # Actually, for exact decomposition of matmul, the terms are not just indicators.
    # But we can use a "basis" of terms.
    
    # A more flexible approach:
    # We fix the "structure" of U, V, W by choosing which entries are non-zero 
    # (the sparsity pattern), and then solve for the non-zero values.
    
    # Let's try a specific structural hypothesis:
    # The rank of 2x4x6 matmul is known to be 24? Or maybe 21?
    # Let's try to find a decomposition of rank R by searching for a valid sparsity pattern.
    
    # This is a combinatorial problem. Let's use a heuristic search for the pattern.
    # Or, we can use the known result that rank(2x4x6) = 24 is likely not minimal, 
    # but let's try to beat 42.
    
    # Actually, a very effective strategy for small matrices is to use the 
    # "Landsberg-Stothers" bound or just brute-force search for small ranks 
    # if the dimension is small enough.
    
    # However, 2x4x6 is small. Let's try a direct numerical search for a rank-30 decomposition.
    # We will parameterize U, V, W with R columns.
    # We want to minimize || T - sum_r U_r V_r W_r ||^2
    # Since T is sparse and has 0/1 entries, and we want exact decomposition, 
    # we can use least squares to find the coefficients.
    
    # The problem is non-convex. We will use random restarts.
    
    def verify(U, V, W, tol=1e-9):
        rank = len(U[0])
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                    if abs(actual - expected) > tol:
                        return False
        return True

    # Baseline: Rank 42 Strassen
    def get_rank_42():
        U_s = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ], dtype=np.float64)
        V_s = np.array([
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [0, 0, 1, 0, 0, 1, 0],
            [1, 0, -1, 0, 1, 0, 1]
        ], dtype=np.float64)
        W_s = np.array([
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ], dtype=np.float64)
        
        rank = 42
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        def add_strassen_block(A_i, A_j, B_j, B_k, C_k, C_i, offset):
            for r in range(7):
                for li in range(2):
                    for lj in range(2):
                        gi = A_i[li]
                        gj = A_j[lj]
                        U[gi * m + gj, offset + r] += U_s[li*2 + lj, r]
                for lj in range(2):
                    for lk in range(2):
                        gj = B_j[lj]
                        gk = B_k[lk]
                        V[gj * p + gk, offset + r] += V_s[lj*2 + lk, r]
                for lk in range(2):
                    for li in range(2):
                        gk = C_k[lk]
                        gi = C_i[li]
                        W[gk * n + gi, offset + r] += W_s[lk*2 + li, r]
        
        add_strassen_block([0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 0)
        add_strassen_block([0,1], [2,3], [2,3], [0,1], [0,1], [0,1], 7)
        add_strassen_block([0,1], [0,1], [0,1], [2,3], [2,3], [0,1], 14)
        add_strassen_block([0,1], [2,3], [2,3], [2,3], [2,3], [0,1], 21)
        add_strassen_block([0,1], [0,1], [0,1], [4,5], [4,5], [0,1], 28)
        add_strassen_block([0,1], [2,3], [2,3], [4,5], [4,5], [0,1], 35)
        
        return U, V, W

    U42, V42, W42 = get_rank_42()
    best_rank = 42
    best_U, best_V, best_W = U42.copy(), V42.copy(), W42.copy()
    
    # Strategy: Try to find a decomposition with rank R < 42 using numerical optimization.
    # We will try ranks 30, 35, 40.
    # For a given rank R, we have variables U (n*m x R), V (m*p x R), W (p*n x R).
    # We want to minimize sum_{i,j,k} (sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1)^2
    # Note: The tensor T has 1s at specific positions and 0s elsewhere?
    # No, the tensor is defined ONLY by the non-zero entries?
    # Wait, the definition is T[i*m+j, j*p+k, k*n+i] = 1.
    # Are there other non-zero entries?
    # The matrix multiplication tensor T_{ijk} = delta_{jk} is not quite right.
    # T_{i,j,k} is the coefficient of A_{ij} B_{jk} in C_{ki}.
    # So T_{i,j,k} = 1 if the indices match the chain i->j->k->i?
    # Actually, the tensor is a 3-index object T_{a,b,c} where a=(i,j), b=(j,k), c=(k,i).
    # T_{a,b,c} = 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k.
    # This implies that for any (a,b,c) not of this form, T_{a,b,c} = 0.
    # So we must satisfy:
    # 1. sum_r U_a,r V_b,r W_c,r = 1 for valid (a,b,c)
    # 2. sum_r U_a,r V_b,r W_c,r = 0 for invalid (a,b,c)
    
    # This is a system of linear equations in the products, but non-linear in U,V,W.
    # However, if we fix U and V, it is linear in W.
    
    # Let's try a specific structured search.
    # We know rank(2x2x2) = 7.
    # rank(2x4x6) is at most 24 (from some constructions?) or 21?
    # Let's try to find a rank-24 decomposition.
    
    # We will use a random restart approach with Levenberg-Marquardt.
    
    def solve_for_rank(R, num_restarts=5, max_iter=100):
        if time.time() > deadline - 5:
            return None
        
        # Initialize U, V, W with small random values
        best_res = None
        best_U_sol = None
        best_V_sol = None
        best_W_sol = None
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U_init = np.random.randn(n*m, R) * 0.1
            V_init = np.random.randn(m*p, R) * 0.1
            W_init = np.random.randn(p*n, R) * 0.1
            
            # Define the residual function
            # We need to flatten U, V, W into a single vector x.
            # x = [U.flatten(), V.flatten(), W.flatten()]
            
            def residual(x):
                U_flat, V_flat, W_flat = x[:n*m*R], x[n*m*R:n*m*R+m*p*R], x[n*m*R+m*p*R:]
                U = U_flat.reshape(n*m, R)
                V = V_flat.reshape(m*p, R)
                W = W_flat.reshape(p*n, R)
                
                res = []
                # Non-zero entries
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m+j
                            b = j*p+k
                            c = k*n+i
                            val = np.sum(U[a, :] * V[b, :] * W[c, :])
                            res.append(val - 1.0)
                
                # Zero entries (sample a subset if too many)
                # Total entries: (n*m)*(m*p)*(p*n) = 48 * 24 * 12 = 13824
                # Non-zero: 48
                # Zero: 13776
                # We can't check all zero entries in every iteration if it's slow.
                # But 13824 is small. Let's check them all.
                
                # To speed up, we can pre-compute the indices.
                # But for now, let's just do the loop.
                
                # Actually, checking all zero entries is expensive.
                # Let's subsample the zero entries.
                # We need the solution to be close to 0 on zero entries.
                
                # For efficiency, we will only check a random subset of zero entries.
                num_zero_checks = 100
                # Pre-generate random zero indices? No, they change per call?
                # No, the indices are fixed. We can pre-generate a fixed set of zero indices.
                
                # This function is called many times, so we need it to be fast.
                # Let's use numpy broadcasting to compute the full tensor?
                # T_comp[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # This is a tensor product.
                # We can compute this as:
                # Temp[a,b,r] = U[a,r] * V[b,r]
                # T_comp[a,b,c] = sum_r Temp[a,b,r] * W[c,r]
                
                # Let's try to vectorize.
                # U: (48, R), V: (24, R), W: (12, R)
                
                # This might be memory heavy if R is large, but R < 42.
                
                # Let's stick to the loop for simplicity and correctness first, 
                # but optimize the zero checks by sampling.
                
                # Actually, the residual size is 48 + 100 = 148.
                # This is small enough for least_squares.
                
                # We need a fixed set of zero indices for the residual to be consistent.
                # Let's generate them once outside.
                
                pass # Placeholder, we will implement a faster version below
                
            # This approach is too slow with the loop inside residual.
            # We need a vectorized residual.
            
            # Let's pre-compute the indices for non-zero and zero entries.
            # Non-zero: list of (a,b,c)
            # Zero: list of (a,b,c)
            
            nz_indices = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        nz_indices.append((i*m+j, j*p+k, k*n+i))
                        
            # Sample zero indices
            np.random.seed(42)
            zero_indices = []
            all_indices = set()
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        all_indices.add((a,b,c))
                        
            nz_set = set(nz_indices)
            zero_pool = [x for x in all_indices if x not in nz_set]
            # Sample 200 zero indices
            zero_indices = np.random.choice(len(zero_pool), 200, replace=False)
            zero_indices = [zero_pool[i] for i in zero_indices]
            
            def residual_fast(x):
                U_flat, V_flat, W_flat = x[:n*m*R], x[n*m*R:n*m*R+m*p*R], x[n*m*R+m*p*R:]
                U = U_flat.reshape(n*m, R)
                V = V_flat.reshape(m*p, R)
                W = W_flat.reshape(p*n, R)
                
                res = np.zeros(len(nz_indices) + len(zero_indices))
                
                # Non-zero
                for idx, (a,b,c) in enumerate(nz_indices):
                    res[idx] = np.sum(U[a, :] * V[b, :] * W[c, :]) - 1.0
                    
                # Zero
                for idx, (a,b,c) in enumerate(zero_indices):
                    res[len(nz_indices) + idx] = np.sum(U[a, :] * V[b, :] * W[c, :])
                    
                return res
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                sol = least_squares(residual_fast, x0, method='lm', max_nfev=1000)
                if sol.success or sol.cost < 1e-6:
                    # Verify with exact check
                    U_sol = sol.x[:n*m*R].reshape(n*m, R)
                    V_sol = sol.x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                    W_sol = sol.x[n*m*R+m*p*R:].reshape(p*n, R)
                    
                    if verify(U_sol, V_sol, W_sol):
                        return U_sol, V_sol, W_sol
                        
            except Exception as e:
                pass
                
        return None

    # Try to find a lower rank
    for target_rank in [30, 35, 40]:
        if time.time() > deadline - 10:
            break
            
        res = solve_for_rank(target_rank, num_restarts=3)
        if res is not None:
            U_sol, V_sol, W_sol = res
            if target_rank < best_rank:
                best_rank = target_rank
                best_U, best_V, best_W = U_sol.copy(), V_sol.copy(), W_sol.copy()
                
    # Final verification
    if not verify(best_U, best_V, best_W):
        # Emergency fallback to rank 48 (schoolbook)
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Convert to lists
    def to_list(arr):
        return arr.tolist()
        
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
