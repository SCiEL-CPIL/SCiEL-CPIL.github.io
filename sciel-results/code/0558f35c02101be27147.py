# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    Uses Vectorized Alternating Least Squares (ALS) with polishing to find low-rank decompositions.
    """
    start_time = time.time()
    deadline = 145  # Keep 5s safety margin before 150s kill
    n, m, p = 3, 4, 5
    nm, mp, pn = n*m, m*p, p*n  # 12, 20, 15
    
    def check_solution(U, V, W, rank):
        """Compute max absolute error."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    val = np.dot(U[u_idx, :rank], V[v_idx, :rank] * W[w_idx, :rank])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def polish(U, V, W, rank):
        """
        Polish the solution by solving for W exactly given U and V,
        then V given U and W, then U given V and W.
        """
        # Solve for W:
        # For each (k, i), we have sum_j U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
        # This is linear in W[k*n+i, :].
        # Let A_w[k*n+i, j, r] = U[i*m+j, r] V[j*p+k, r]
        # We want sum_j sum_r A_w[k*n+i, j, r] W[k*n+i, r] = 1
        # This is not quite right. The term is sum_r (U[i*m+j, r] V[j*p+k, r]) W[k*n+i, r].
        # So for fixed (k, i), the equation is:
        # sum_j (U[i*m+j, :] * V[j*p+k, :])^T W[k*n+i, :] = 1
        # Let B[k*n+i, j, r] = U[i*m+j, r] * V[j*p+k, r]
        # Then sum_j B[k*n+i, j, r] W[k*n+i, r] = 1
        # This is a linear system for W[k*n+i, :] of size rank.
        # The coefficient matrix is M_w[k*n+i, r, s] = sum_j B[k*n+i, j, r] * delta(r,s)? No.
        # The equation is: sum_j sum_r B[k*n+i, j, r] W[k*n+i, r] = 1.
        # Wait, W is a vector of length rank.
        # The term is sum_r (sum_j B[k*n+i, j, r]) W[k*n+i, r] = 1.
        # This is a single equation for each (k,i). But W[k*n+i, :] has rank components.
        # We have pn equations (one for each k,i) and pn*rank unknowns? No.
        # The decomposition is:
        # 1 = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # For fixed (k,i), summing over j is not part of the tensor definition.
        # The tensor entry T[i,j,k] = 1.
        # The decomposition approximates T[i,j,k] by sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
        # So for each (i,j,k), we have:
        # sum_r (U[i*m+j, r] V[j*p+k, r]) W[k*n+i, r] = 1.
        # If we fix U and V, this is a linear system for W.
        # Unknowns: W[k*n+i, r] for all k,i,r.
        # Equations: One for each (i,j,k). Total nm*p = 60 equations.
        # Unknowns: pn * rank = 15 * rank.
        # If rank < 4, we have more unknowns than equations? 15*rank > 60 => rank > 4.
        # So it's an underdetermined system if rank is large. We want the minimum norm solution or just any solution.
        # We can use np.linalg.lstsq.
        
        # Construct the matrix A_w of shape (60, pn*rank)
        # Row index: idx = (i*p + j)*p + k ? No, let's map (i,j,k) to a flat index.
        # Let's just loop and build the system.
        
        A_w = np.zeros((n*m*p, pn*rank))
        b_w = np.ones(n*m*p)
        
        row = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    w_row_base = k*n + i
                    for r in range(rank):
                        A_w[row, w_row_base*rank + r] = U[i*m+j, r] * V[j*p+k, r]
                    row += 1
        
        try:
            W_flat, residuals, rank_w, s = np.linalg.lstsq(A_w, b_w, rcond=None)
            W_new = W_flat.reshape(pn, rank)
        except:
            W_new = W
            
        # Similarly for V and U?
        # Polishing W is enough to reduce error significantly if U and V are good.
        return W_new

    def solve_als(rank, seed, max_iters=200, time_budget=5.0):
        """
        Solve for U, V, W using ALS.
        """
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(nm, rank)
        V = rng.randn(mp, rank)
        W = rng.randn(pn, rank)
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() - start_time > time_budget + start_time:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    u_idx = i*m + j
                    # A_u[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # b_u[k] = 1
                    A_u = np.zeros((p, rank))
                    for k in range(p):
                        v_idx = j*p + k
                        w_idx = k*n + i
                        A_u[k, :] = V[v_idx, :] * W[w_idx, :]
                    b_u = np.ones(p)
                    try:
                        U[u_idx, :], _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                    except:
                        pass
                        
            # Update V
            for j in range(m):
                for k in range(p):
                    v_idx = j*p + k
                    A_v = np.zeros((n, rank))
                    for i in range(n):
                        u_idx = i*m + j
                        w_idx = k*n + i
                        A_v[i, :] = U[u_idx, :] * W[w_idx, :]
                    b_v = np.ones(n)
                    try:
                        V[v_idx, :], _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                    except:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    w_idx = k*n + i
                    A_w = np.zeros((m, rank))
                    for j in range(m):
                        u_idx = i*m + j
                        v_idx = j*p + k
                        A_w[j, :] = U[u_idx, :] * V[v_idx, :]
                    b_w = np.ones(m)
                    try:
                        W[w_idx, :], _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
                    except:
                        pass
            
            # Check error
            curr_err = check_solution(U, V, W, rank)
            if curr_err < best_err:
                best_err = curr_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-10:
                break
                
        # Polish
        W_polished = polish(best_U, best_V, best_W, rank)
        final_err = check_solution(best_U, best_V, W_polished, rank)
        if final_err < best_err:
            best_err = final_err
            best_W = W_polished

        return best_err, best_U, best_V, best_W

    # Schoolbook baseline
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Search ranks
    # We want the minimum rank.
    # Try ranks from 30 to 60.
    # If we find a solution at rank R, we should try to find a lower rank.
    # Strategy:
    # 1. Try ranks in order: 30, 31, ..., 60.
    # 2. If we find a solution at R, set best_rank = R.
    # 3. Continue to check lower ranks? No, we are going up.
    # 4. So if we find 40, we have to check 30-39.
    # 5. This is inefficient if the optimal rank is high.
    # 6. Better: Try a set of candidate ranks.
    
    candidates = [30, 35, 40, 45, 50, 55, 58, 59, 60]
    
    for rank in candidates:
        if time.time() - start_time > deadline - 10:
            break
            
        if rank >= best_rank:
            continue
            
        # Try multiple seeds
        found = False
        for seed in [42, 123, 999, 2024, 7, 11, 13, 17, 19, 23]:
            if time.time() - start_time > deadline - 10:
                break
                
            time_left = deadline - (time.time() - start_time) - 5
            if time_left < 2:
                break
            
            err, U_c, V_c, W_c = solve_als(rank, seed, max_iters=100, time_budget=time_left/10.0)
            
            if err < 1e-8:
                best_rank = rank
                best_U, best_V, best_W = U_c, V_c, W_c
                found = True
                break
        
        if found:
            # If we found a solution, we should try to see if a lower rank is possible.
            # But we are iterating up. So we should continue the loop to check lower ranks?
            # No, the loop goes 30, 35, 40...
            # If we found 40, we have already checked 30, 35.
            # So we can break if we found a solution at the lowest candidate?
            # No, we might find 40, but 38 might work.
            # Since we skipped 31-34, 36-39, etc., we might miss the minimum.
            # To be safe, if we find a solution at R, we should try R-1, R-2... down to the previous candidate.
            # Let's modify the loop to be more adaptive.
            pass

    # Adaptive search: if we found a solution at R, search R-1 down to 30.
    if best_rank < rank_schoolbook:
        current_best = best_rank
        for r in range(current_best - 1, 29, -1):
            if time.time() - start_time > deadline - 5:
                break
            
            # Try to find solution at rank r
            found_lower = False
            for seed in [42, 123, 999]:
                if time.time() - start_time > deadline - 5:
                    break
                time_left = deadline - (time.time() - start_time) - 2
                if time_left < 1:
                    break
                err, U_c, V_c, W_c = solve_als(r, seed, max_iters=50, time_budget=time_left/3.0)
                if err < 1e-8:
                    best_rank = r
                    best_U, best_V, best_W = U_c, V_c, W_c
                    found_lower = True
                    break
            
            if found_lower:
                continue
            else:
                # If we fail to find a solution at r, we might assume r is not the minimum,
                # but it could be due to local minima.
                # We'll just stop the downward search if we fail, to save time.
                break

    # Final verification
    if best_rank < rank_schoolbook:
        final_err = check_solution(best_U, best_V, best_W, best_rank)
        if final_err > 1e-6:
            # Fallback
            best_rank = rank_schoolbook
            best_U, best_V, best_W = U_sb, V_sb, W_sb

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    U_out = to_list(best_U[:, :best_rank])
    V_out = to_list(best_V[:, :best_rank])
    W_out = to_list(best_W[:, :best_rank])
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
