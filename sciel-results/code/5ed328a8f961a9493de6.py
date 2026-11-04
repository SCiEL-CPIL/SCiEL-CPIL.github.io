<<<<<<< SEARCH
def _try_rank16_search(start_time, deadline, n, m, p, n_a, n_b, n_c):
    """
    Attempt to find a rank-16 decomposition using a structured ansatz.
    
    Strategy: Start from the rank-17 Strassen-based decomposition and try to
    merge/eliminate one rank term by solving for a 16-term decomposition.
    
    We use a parameterized search: fix the structure similar to the rank-17
    solution but with 16 terms, and solve the resulting nonlinear system.
    """
    import random
    from fractions import Fraction
    
    T = _build_tensor(n, m, p)
    rank_target = 16
    
    # Try a structured approach: use the known rank-7 for 2x2x2 (Strassen)
    # and try to handle the remaining 10 terms (2x2x1 + 2x1x3) with 9 terms
    # instead of 10, saving 1 rank.
    #
    # The "remaining" part involves:
    # - A[:,2] (column 2 of A, i.e., a2, a5) times B[2,:] (row 2 of B, i.e., b6,b7,b8)
    # - A[0:2,0:2] (top-left 2x2) times B[0:2,2] (right column of top-left 2x2, i.e., b2,b5)
    #
    # Total terms needed: 6 (for a2,b6) + 6 (for a5,b7) + 6 (for a5,b8) 
    #   + 4 (for a0,b2) + 4 (for a1,b5) + 4 (for a3,b2) + 4 (for a4,b5)
    # But many of these can be combined.
    #
    # Let's try a randomized search with structured initialization.
    
    rng = np.random.RandomState(42)
    
    # Try multiple random initializations
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        
        # Initialize with random values, but with some structure
        # Use the rank-17 solution as a starting point and try to zero out one column
        U = np.zeros((n_a, rank_target))
        V = np.zeros((n_b, rank_target))
        W = np.zeros((n_c, rank_target))
        
        # Start from the rank-17 solution, drop one column, and adjust
        # First build the rank-17 solution
        U17 = np.zeros((n_a, 17))
        V17 = np.zeros((n_b, 17))
        W17 = np.zeros((n_c, 17))
        
        def set_term(r, u_dict, v_dict, w_dict, U_mat, V_mat, W_mat):
            for a, val in u_dict.items():
                U_mat[a, r] = val
            for b, val in v_dict.items():
                V_mat[b, r] = val
            for c, val in w_dict.items():
                W_mat[c, r] = val
        
        set_term(0, {0:1, 4:1}, {0:1, 4:1}, {0:1, 3:1}, U17, V17, W17)
        set_term(1, {3:1, 4:1}, {0:1}, {1:1, 3:-1}, U17, V17, W17)
        set_term(2, {0:1}, {1:1, 4:-1}, {2:1, 3:1}, U17, V17, W17)
        set_term(3, {4:1}, {3:1, 0:-1}, {0:1, 1:1}, U17, V17, W17)
        set_term(4, {0:1, 1:1}, {4:1}, {0:-1, 2:1}, U17, V17, W17)
        set_term(5, {3:1, 0:-1}, {0:1, 1:1}, {3:1}, U17, V17, W17)
        set_term(6, {1:1, 4:-1}, {3:1, 4:1}, {0:1}, U17, V17, W17)
        set_term(7, {0:1}, {2:1}, {4:1}, U17, V17, W17)
        set_term(8, {1:1}, {5:1}, {4:1}, U17, V17, W17)
        set_term(9, {3:1}, {2:1}, {5:1}, U17, V17, W17)
        set_term(10, {4:1}, {5:1}, {5:1}, U17, V17, W17)
        set_term(11, {2:1}, {6:1}, {0:1}, U17, V17, W17)
        set_term(12, {2:1}, {7:1}, {2:1}, U17, V17, W17)
        set_term(13, {2:1}, {8:1}, {4:1}, U17, V17, W17)
        set_term(14, {5:1}, {6:1}, {1:1}, U17, V17, W17)
        set_term(15, {5:1}, {7:1}, {3:1}, U17, V17, W17)
        set_term(16, {5:1}, {8:1}, {5:1}, U17, V17, W17)
        
        # Try dropping each column and see if we can adjust the remaining 16
        drop_col = attempt % 17
        if drop_col >= 16:
            drop_col = 15
        
        # Keep columns 0..15, dropping column drop_col
        keep_cols = [c for c in range(17) if c != drop_col]
        U_init = U17[:, keep_cols].copy()
        V_init = V17[:, keep_cols].copy()
        W_init = W17[:, keep_cols].copy()
        
        # Add small random perturbation
        scale = 0.01
        U_init += rng.randn(n_a, rank_target) * scale
        V_init += rng.randn(n_b, rank_target) * scale
        W_init += rng.randn(n_c, rank_target) * scale
        
        # Now try to solve using iterative refinement
        # We want: sum_r U[:,r] V[:,r] W[:,r] = T
        # This is equivalent to: for each (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        
        # Use a simple iterative method: alternate between fixing two matrices
        # and solving for the third.
        
        U_cur = U_init.copy()
        V_cur = V_init.copy()
        W_cur = W_init.copy()
        
        # Precompute T as a flat array for faster access
        T_flat = T.flatten()
        
        # Number of equations: n_a * n_b * n_c = 6*9*6 = 324
        # Number of unknowns: 3 * rank * (n_a + n_b + n_c) but structured
        # Actually we have rank*(n_a + n_b + n_c) = 16*(6+9+6) = 336 unknowns
        
        # Use a Gauss-Seidel style iteration
        max_iters = 200
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # For each rank term r, try to update
            for r in range(rank_target):
                # Compute residual contribution from other terms
                # R[a,b,c] = T[a,b,c] - sum_{r' != r} U[a,r']*V[b,r']*W[c,r']
                
                # This is expensive to compute fully. Instead, use a gradient-free approach.
                # Try a simple random walk with acceptance based on error reduction.
                pass
        
        # Alternative: use scipy least squares on a linearized problem
        # Linearize around current solution
        # Actually, let's try a different approach: use the fact that we can
        # express this as a tensor network and use ALS (Alternating Least Squares)
        
        # ALS approach: fix U and V, solve for W, then fix U and W, solve for V, etc.
        # For fixed U and V, the equation for W is:
        # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        # For each c, this gives a linear system in W[:,c]
        
        # Actually, let's use a simpler approach: 
        # The tensor T can be written as sum_r u_r * v_r * w_r (outer products)
        # We want to find rank-16 decomposition.
        
        # Use a randomized SVD-like approach on the matricization
        # Matricize T as a (n_a * n_b) x n_c matrix
        T_mat = T.reshape(n_a * n_b, n_c)
        
        # Try to find low-rank approximation
        # But this won't work directly because we need the specific structure
        
        # Let's try a more direct approach: 
        # Use the known border rank decomposition of 3x3x3 (rank 10)
        # and adapt it to 2x3x3.
        
        # For now, return None to indicate failure
        continue
    
    return None
=======
def _try_rank16_search(start_time, deadline, n, m, p, n_a, n_b, n_c):
    """
    Attempt to find a rank-16 decomposition using ALS (Alternating Least Squares).
    
    Strategy: Initialize from the rank-17 solution by dropping one column,
    then use ALS to refine the remaining 16 terms.
    """
    T = _build_tensor(n, m, p)
    rank_target = 16
    rng = np.random.RandomState(42)
    
    # Build the rank-17 solution
    U17 = np.zeros((n_a, 17))
    V17 = np.zeros((n_b, 17))
    W17 = np.zeros((n_c, 17))
    
    def set_term(r, u_dict, v_dict, w_dict, U_mat, V_mat, W_mat):
        for a, val in u_dict.items():
            U_mat[a, r] = val
        for b, val in v_dict.items():
            V_mat[b, r] = val
        for c, val in w_dict.items():
            W_mat[c, r] = val
    
    set_term(0, {0:1, 4:1}, {0:1, 4:1}, {0:1, 3:1}, U17, V17, W17)
    set_term(1, {3:1, 4:1}, {0:1}, {1:1, 3:-1}, U17, V17, W17)
    set_term(2, {0:1}, {1:1, 4:-1}, {2:1, 3:1}, U17, V17, W17)
    set_term(3, {4:1}, {3:1, 0:-1}, {0:1, 1:1}, U17, V17, W17)
    set_term(4, {0:1, 1:1}, {4:1}, {0:-1, 2:1}, U17, V17, W17)
    set_term(5, {3:1, 0:-1}, {0:1, 1:1}, {3:1}, U17, V17, W17)
    set_term(6, {1:1, 4:-1}, {3:1, 4:1}, {0:1}, U17, V17, W17)
    set_term(7, {0:1}, {2:1}, {4:1}, U17, V17, W17)
    set_term(8, {1:1}, {5:1}, {4:1}, U17, V17, W17)
    set_term(9, {3:1}, {2:1}, {5:1}, U17, V17, W17)
    set_term(10, {4:1}, {5:1}, {5:1}, U17, V17, W17)
    set_term(11, {2:1}, {6:1}, {0:1}, U17, V17, W17)
    set_term(12, {2:1}, {7:1}, {2:1}, U17, V17, W17)
    set_term(13, {2:1}, {8:1}, {4:1}, U17, V17, W17)
    set_term(14, {5:1}, {6:1}, {1:1}, U17, V17, W17)
    set_term(15, {5:1}, {7:1}, {3:1}, U17, V17, W17)
    set_term(16, {5:1}, {8:1}, {5:1}, U17, V17, W17)
    
    # Try dropping each of the 17 columns and run ALS
    for drop_col in range(17):
        if time.time() > deadline - 10:
            break
        
        keep_cols = [c for c in range(17) if c != drop_col]
        U = U17[:, keep_cols].copy()
        V = V17[:, keep_cols].copy()
        W = W17[:, keep_cols].copy()
        
        # Add small perturbation to avoid getting stuck
        U += rng.randn(n_a, rank_target) * 0.01
        V += rng.randn(n_b, rank_target) * 0.01
        W += rng.randn(n_c, rank_target) * 0.01
        
        # ALS: alternate between solving for U, V, W
        # For fixed V and W, solve for U:
        #   For each a: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all b,c
        #   This is a linear system: U[a,:] * (V[:,r] * W[:,r] outer product) = T[a,:,:]
        #   Specifically: U[a,r] = sum_{b,c} T[a,b,c] * V[b,r] * W[c,r] / sum_{b,c} V[b,r]^2 * W[c,r]^2
        
        # Actually, the correct ALS update for U given V,W is:
        # U[a,r] = <T[a,:,:], V[:,r] * W[:,r]> / ||V[:,r] * W[:,r]||^2
        # where V[:,r] * W[:,r] is the element-wise product (outer product matricized)
        
        max_als_iters = 50
        for als_iter in range(max_als_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U
            for a in range(n_a):
                for r in range(rank_target):
                    # Compute numerator: sum_{b,c} T[a,b,c] * V[b,r] * W[c,r]
                    num = np.sum(T[a, :, :] * np.outer(V[:, r], W[:, r]).T)
                    # Compute denominator: sum_{b,c} V[b,r]^2 * W[c,r]^2
                    den = np.sum(V[:, r]**2) * np.sum(W[:, r]**2)
                    if den > 1e-12:
                        U[a, r] = num / den
            
            # Update V
            for b in range(n_b):
                for r in range(rank_target):
                    num = np.sum(T[:, b, :] * np.outer(U[:, r], W[:, r]).T)
                    den = np.sum(U[:, r]**2) * np.sum(W[:, r]**2)
                    if den > 1e-12:
                        V[b, r] = num / den
            
            # Update W
            for c in range(n_c):
                for r in range(rank_target):
                    num = np.sum(T[:, :, c] * np.outer(U[:, r], V[:, r]).T)
                    den = np.sum(U[:, r]**2) * np.sum(V[:, r]**2)
                    if den > 1e-12:
                        W[c, r] = num / den
            
            # Check convergence
            if als_iter % 10 == 9:
                # Compute error
                result = np.zeros(T.shape)
                for r in range(rank_target):
                    result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                err = np.max(np.abs(result - T))
                if err < 1e-8:
                    # Found a valid rank-16 decomposition!
                    # Round to rational values if possible
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Try with a different perturbation scale
        U = U17[:, keep_cols].copy()
        V = V17[:, keep_cols].copy()
        W = W17[:, keep_cols].copy()
        U += rng.randn(n_a, rank_target) * 0.1
        V += rng.randn(n_b, rank_target) * 0.1
        W += rng.randn(n_c, rank_target) * 0.1
        
        for als_iter in range(max_als_iters):
            if time.time() > deadline - 5:
                break
            
            for a in range(n_a):
                for r in range(rank_target):
                    num = np.sum(T[a, :, :] * np.outer(V[:, r], W[:, r]).T)
                    den = np.sum(V[:, r]**2) * np.sum(W[:, r]**2)
                    if den > 1e-12:
                        U[a, r] = num / den
            
            for b in range(n_b):
                for r in range(rank_target):
                    num = np.sum(T[:, b, :] * np.outer(U[:, r], W[:, r]).T)
                    den = np.sum(U[:, r]**2) * np.sum(W[:, r]**2)
                    if den > 1e-12:
                        V[b, r] = num / den
            
            for c in range(n_c):
                for r in range(rank_target):
                    num = np.sum(T[:, :, c] * np.outer(U[:, r], V[:, r]).T)
                    den = np.sum(U[:, r]**2) * np.sum(V[:, r]**2)
                    if den > 1e-12:
                        W[c, r] = num / den
            
            if als_iter % 10 == 9:
                result = np.zeros(T.shape)
                for r in range(rank_target):
                    result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                err = np.max(np.abs(result - T))
                if err < 1e-8:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None
>>>>>>> REPLACE
