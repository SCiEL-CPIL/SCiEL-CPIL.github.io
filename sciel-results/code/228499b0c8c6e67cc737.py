import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    n, m, p = 2, 2, 11
    # Dimensions of the tensor
    dim_a = n * m  # 4
    dim_b = m * p  # 22
    dim_c = p * n  # 22
    
    # Build the multiplication tensor T of shape (dim_a, dim_b, dim_c)
    # T[a_idx, b_idx, c_idx] = 1 if the indices are consistent
    # a_idx = i*m + j (i in 0..n-1, j in 0..m-1)
    # b_idx = j*p + k (j in 0..m-1, k in 0..p-1)
    # c_idx = k*n + i (k in 0..p-1, i in 0..n-1)
    
    T = np.zeros((dim_a, dim_b, dim_c), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    def tensor_rank_estimate(T_tensor):
        """Estimate the tensor rank by checking if it can be decomposed with r terms."""
        # This is NP-hard in general, so we use a heuristic
        pass
    
    def verify_decomposition(U, V, W, T_target, tol=1e-9):
        """Verify that U, V, W exactly decompose T_target."""
        # T_approx[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Efficient: for each r, outer product
        T_approx = np.zeros_like(T_target)
        for r in range(U.shape[1]):
            T_approx += np.einsum('ar,br,cr->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(T_approx, T_target, atol=tol)
    
    def try_reduce_rank(U, V, W, T_target, max_iterations=2000, tol=1e-9):
        """
        Try to reduce the rank by finding linear dependencies.
        Strategy: repeatedly try to remove a rank-1 term by expressing it
        as a linear combination of others, or merge terms.
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        current_rank = U.shape[1]
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            if current_rank <= 0:
                break
            
            # Try to find a column that is a linear combination of others
            # Check each column r: can U[:,r] be expressed as combo of other U columns?
            # And simultaneously for V and W?
            
            found_reduction = False
            for r in range(current_rank):
                if time.time() > deadline:
                    break
                
                # Check if removing column r still allows exact decomposition
                # by solving for a correction in the remaining columns
                if current_rank <= 1:
                    # Can't reduce below 1 if tensor is nonzero
                    if current_rank == 1 and np.any(T_target != 0):
                        continue
                    else:
                        # Remove it
                        U = np.delete(U, r, axis=1)
                        V = np.delete(V, r, axis=1)
                        W = np.delete(W, r, axis=1)
                        current_rank -= 1
                        found_reduction = True
                        break
                    continue
                
                # Try: can we express term r as linear combo of other terms?
                # We need: U[:,r] = U_others @ alpha, V[:,r] = V_others @ beta, W[:,r] = W_others @ gamma
                # where alpha, beta, gamma are the same vector (since it's a single rank-1 term being decomposed)
                # Actually no - we need the PRODUCT to match.
                # 
                # Better approach: check if the tensor residual after removing term r
                # can be absorbed into the remaining terms.
                
                # Alternative: use the fact that if the matrix flattening has rank deficiency,
                # we might be able to reduce.
                
                # Let's try a different approach: check if two terms can be merged
                # or if a term is redundant.
                
                # Check if U[:,r] is in span of U[:, others]
                U_others = np.delete(U, r, axis=1)
                V_others = np.delete(V, r, axis=1)
                W_others = np.delete(W, r, axis=1)
                
                # Check if U[:,r] is in column space of U_others
                # Use least squares
                if U_others.shape[1] > 0:
                    # Solve U_others @ x ≈ U[:,r]
                    try:
                        x_u, residuals_u, rank_u, _ = np.linalg.lstsq(U_others, U[:, r], rcond=None)
                        u_residual = U[:, r] - U_others @ x_u
                        u_residual_norm = np.linalg.norm(u_residual)
                        
                        if u_residual_norm < tol:
                            # U[:,r] is in span, check V
                            x_v, residuals_v, rank_v, _ = np.linalg.lstsq(V_others, V[:, r], rcond=None)
                            v_residual = V[:, r] - V_others @ x_v
                            v_residual_norm = np.linalg.norm(v_residual)
                            
                            if v_residual_norm < tol:
                                # Check W
                                x_w, residuals_w, rank_w, _ = np.linalg.lstsq(W_others, W[:, r], rcond=None)
                                w_residual = W[:, r] - W_others @ x_w
                                w_residual_norm = np.linalg.norm(w_residual)
                                
                                if w_residual_norm < tol:
                                    # All three are in span - but we need the SAME coefficients
                                    # Actually, for the tensor to be preserved, we need:
                                    # U[:,r]*V[:,r]*W[:,r] = sum_s (U_others[:,s]*x_u[s]) * (V_others[:,s]*x_v[s]) * (W_others[:,s]*x_w[s])
                                    # This is NOT generally true unless x_u = x_v = x_w
                                    # 
                                    # Better: just remove the term and check if the tensor still matches
                                    U_test = np.delete(U, r, axis=1)
                                    V_test = np.delete(V, r, axis=1)
                                    W_test = np.delete(W, r, axis=1)
                                    
                                    T_test = np.zeros_like(T_target)
                                    for s in range(U_test.shape[1]):
                                        T_test += np.einsum('as,bs,cs->abc', U_test[:, s], V_test[:, s], W_test[:, s])
                                    
                                    if np.allclose(T_test, T_target, atol=tol):
                                        U = U_test
                                        V = V_test
                                        W = W_test
                                        current_rank -= 1
                                        found_reduction = True
                                        break
                    except:
                        pass
            
            if not found_reduction:
                break
        
        return U, V, W, current_rank
    
    # Start with the known 39-rank decomposition from Strassen
    rank_init = 39
    U = np.zeros((dim_a, rank_init))
    V = np.zeros((dim_b, rank_init))
    W = np.zeros((dim_c, rank_init))
    
    r = 0
    for block in range(5):
        k_offset = block * 2
        v_e = k_offset
        v_f = k_offset + 1
        v_g = p + k_offset
        v_h = p + k_offset + 1
        w_c00 = 2 * k_offset
        w_c10 = 2 * k_offset + 1
        w_c01 = 2 * k_offset + 2
        w_c11 = 2 * k_offset + 3
        
        # m1 = (a+d)(e+h) -> c00 + c11
        U[0, r] += 1; U[3, r] += 1
        V[v_e, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1; W[w_c11, r] += 1
        r += 1
        
        # m2 = (c+d)e -> c10 - c11
        U[2, r] += 1; U[3, r] += 1
        V[v_e, r] += 1
        W[w_c10, r] += 1; W[w_c11, r] += -1
        r += 1
        
        # m3 = a(f-h) -> c01 + c11
        U[0, r] += 1
        V[v_f, r] += 1; V[v_h, r] += -1
        W[w_c01, r] += 1; W[w_c11, r] += 1
        r += 1
        
        # m4 = d(g-e) -> c00 + c10
        U[3, r] += 1
        V[v_g, r] += 1; V[v_e, r] += -1
        W[w_c00, r] += 1; W[w_c10, r] += 1
        r += 1
        
        # m5 = (a+b)h -> -c00 + c01
        U[0, r] += 1; U[1, r] += 1
        V[v_h, r] += 1
        W[w_c00, r] += -1; W[w_c01, r] += 1
        r += 1
        
        # m6 = (c-a)(e+f) -> c11
        U[2, r] += 1; U[0, r] += -1
        V[v_e, r] += 1; V[v_f, r] += 1
        W[w_c11, r] += 1
        r += 1
        
        # m7 = (b-d)(g+h) -> c00
        U[1, r] += 1; U[3, r] += -1
        V[v_g, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1
        r += 1
    
    # Last column schoolbook
    k2 = 10
    v_e2 = k2
    v_g2 = p + k2
    w_c02 = 2 * k2
    w_c12 = 2 * k2 + 1
    
    U[0, r] += 1; V[v_e2, r] += 1; W[w_c02, r] += 1; r += 1
    U[1, r] += 1; V[v_g2, r] += 1; W[w_c02, r] += 1; r += 1
    U[2, r] += 1; V[v_e2, r] += 1; W[w_c12, r] += 1; r += 1
    U[3, r] += 1; V[v_g2, r] += 1; W[w_c12, r] += 1; r += 1
    
    # Verify initial decomposition
    assert verify_decomposition(U, V, W, T), "Initial decomposition is invalid"
    
    # Now try to reduce rank
    U_red, V_red, W_red, reduced_rank = try_reduce_rank(U, V, W, T)
    
    if reduced_rank < rank_init:
        U, V, W = U_red, V_red, W_red
        rank_init = reduced_rank
    
    # Try a different structural approach: use a 2x2x2 Strassen-like scheme 
    # but with a different grouping to potentially save more ranks.
    # 
    # Alternative: try to use a "batched" approach where we process 
    # multiple columns of B together with shared structure.
    
    # Try: group columns into blocks of 4 and use a 4-term scheme
    # For a 2x2 * 2x4 block, schoolbook is 16, Strassen-per-2cols is 14
    # Can we do better?
    
    # Let's try a random search for a lower-rank decomposition
    # by starting from random initializations and doing gradient-free optimization
    
    def random_decomposition(rank, seed=42):
        rng = np.random.RandomState(seed)
        U_rand = rng.randn(dim_a, rank)
        V_rand = rng.randn(dim_b, rank)
        W_rand = rng.randn(dim_c, rank)
        return U_rand, V_rand, W_rand
    
    def optimize_decomposition(U_init, V_init, W_init, T_target, max_iter=5000, tol=1e-10):
        """
        Use a simple iterative approach to adjust a rank-r decomposition
        to exactly match the target tensor.
        Uses alternating least squares on the tensor.
        """
        U = np.array(U_init, dtype=float)
        V = np.array(V_init, dtype=float)
        W = np.array(W_init, dtype=float)
        rank = U.shape[1]
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U: for each a, U[a,:] should satisfy
            # sum_r U[a,r] * V[:,r] * W[:,r] = T[a,:,:]
            # Reshape: T_a = T[a,:,:] (shape dim_b x dim_c)
            # We need: sum_r U[a,r] * outer(V[:,r], W[:,r]) = T_a
            # This is a linear system in U[a,:]
            # M_U[a] is a matrix of shape (dim_b*dim_c, rank)
            # where M_U[a][:, r] = V[:,r] * W[:,r] (flattened)
            
            # Precompute M_U, M_V, M_W
            # M_U[:, r] = (V[:,r] * W[:,r]).flatten()
            # M_V[:, r] = (U[:,r] * W[:,r]).flatten()
            # M_W[:, r] = (U[:,r] * V[:,r]).flatten()
            
            # Update U
            for a in range(dim_a):
                # M_U_row is (dim_b * dim_c, rank)
                M_U = np.zeros((dim_b * dim_c, rank))
                for r in range(rank):
                    M_U[:, r] = (V[:, r] * W[:, r]).flatten()
                b_vec = T_target[a, :, :].flatten()
                try:
                    U[a, :], _, _, _ = np.linalg.lstsq(M_U, b_vec, rcond=None)
                except:
                    pass
            
            # Update V
            for b in range(dim_b):
                M_V = np.zeros((dim_a * dim_c, rank))
                for r in range(rank):
                    M_V[:, r] = (U[:, r] * W[:, r]).flatten()
                b_vec = T_target[:, b, :].flatten()
                try:
                    V[b, :], _, _, _ = np.linalg.lstsq(M_V, b_vec, rcond=None)
                except:
                    pass
            
            # Update W
            for c in range(dim_c):
                M_W = np.zeros((dim_a * dim_b, rank))
                for r in range(rank):
                    M_W[:, r] = (U[:, r] * V[:, r]).flatten()
                b_vec = T_target[:, :, c].flatten()
                try:
                    W[c, :], _, _, _ = np.linalg.lstsq(M_W, b_vec, rcond=None)
                except:
                    pass
            
            # Check convergence
            T_approx = np.zeros_like(T_target)
            for r in range(rank):
                T_approx += np.einsum('ar,br,cr->abc', U[:, r], V[:, r], W[:, r])
            err = np.linalg.norm(T_approx - T_target)
            if err < tol:
                break
        
        return U, V, W, err
    
    # Try to find a lower-rank decomposition via random initialization + ALS
    best_rank = rank_init
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try ranks from 38 down
    for target_rank in range(rank_init - 1, 30, -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            
            U_rand, V_rand, W_rand = random_decomposition(target_rank, seed=seed + 100)
            
            U_opt, V_opt, W_opt, err = optimize_decomposition(U_rand, V_rand, W_rand, T, max_iter=3000)
            
            if err < 1e-8:
                # Check exactness with rational-like verification
                if verify_decomposition(U_opt, V_opt, W_opt, T, tol=1e-9):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    break
        
        if best_rank == target_rank:
            break
    
    # Final: convert to the required format
    final_rank = best_rank
    U_final = [[best_U[i, r] for r in range(final_rank)] for i in range(dim_a)]
    V_final = [[best_V[i, r] for r in range(final_rank)] for i in range(dim_b)]
    W_final = [[best_W[i, r] for r in range(final_rank)] for i in range(dim_c)]
    
    # Verify one more time
    U_np = np.array(U_final, dtype=float)
    V_np = np.array(V_final, dtype=float)
    W_np = np.array(W_final, dtype=float)
    if not verify_decomposition(U_np, V_np, W_np, T, tol=1e-8):
        # Fallback to the known 39-rank solution
        U_final = [[U[i, r] for r in range(39)] for i in range(dim_a)]
        V_final = [[V[i, r] for r in range(39)] for i in range(dim_b)]
        W_final = [[W[i, r] for r in range(39)] for i in range(dim_c)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
