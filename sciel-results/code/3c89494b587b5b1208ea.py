import numpy as np
import time
import sys
import scipy.optimize as opt

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    n, m, p = 4, 4, 7
    
    def a_idx(i, j):
        return i * m + j

    def b_idx(j, k):
        return j * p + k

    def c_idx(k, i):
        return k * n + i

    # Precompute the list of active (a, b, c) indices for the target tensor
    # M[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    active_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = a_idx(i, j)
                b = b_idx(j, k)
                c = c_idx(k, i)
                active_indices.append((a, b, c))
                
    # Also precompute a set of inactive indices to check for spurious non-zeros
    # We will check a random subset during verification
    all_a = np.arange(n*m)
    all_b = np.arange(m*p)
    all_c = np.arange(p*n)
    
    def get_inactive_sample(rng, count=100):
        inactive = []
        active_set = set(active_indices)
        while len(inactive) < count:
            a = rng.integers(0, n*m)
            b = rng.integers(0, m*p)
            c = rng.integers(0, p*n)
            if (a, b, c) not in active_set:
                inactive.append((a, b, c))
        return inactive

    def compute_error(U, V, W, inactive_sample=None):
        """
        Compute the error of the decomposition.
        Error = sum_{active} (M_abc - sum_r U_ar V_br W_cr)^2 
              + sum_{inactive} (sum_r U_ar V_br W_cr)^2
        """
        # Vectorized computation for active terms
        # U: (16, R), V: (28, R), W: (28, R)
        # We want to compute T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        
        # For efficiency in the optimizer, we only check a subset of inactive terms
        # because the full tensor is small (16*28*28 = 12544), but we want speed.
        
        # Compute full tensor? 12544 floats is small. Let's just compute it all.
        # T = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        # This is (16, 28, 28). 
        # However, einsum in a tight loop might be slow if called thousands of times.
        # Let's try to be smarter.
        
        # Actually, for L-BFGS, we need gradients. 
        # But we are using finite difference or analytical? scipy.optimize.minimize uses analytical if provided, else finite diff.
        # Finite diff on 16*R + 28*R + 28*R parameters is expensive.
        # Let's provide a Jacobian? Or just use a robust method.
        
        # Alternative: Use the fact that the problem is linear in each factor.
        # But for a single scalar error function, let's just compute the full tensor.
        
        T = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        
        err = 0.0
        for a, b, c in active_indices:
            val = T[a, b, c]
            err += (val - 1.0) ** 2
            
        # Check inactive terms
        # To save time, we can sample. But for correctness, we should check all?
        # 12544 - 112 = 12432 inactive terms.
        # Looping 12432 times in Python is slow (~1ms). 
        # Let's vectorize the inactive check.
        
        # Create a mask of active indices
        # T is (16, 28, 28)
        # We want sum of T[~active]^2
        
        # Instead of building a full mask every time, we can subtract the active sum from the total sum of squares?
        # No, because the target for active is 1, not 0.
        # Total Error = sum_active (T-1)^2 + sum_inactive T^2
        #             = sum_all T^2 - 2*sum_active T + sum_active 1 + sum_inactive T^2
        #             = sum_all T^2 - 2*sum_active T + 112 + (sum_all T^2 - sum_active T^2)
        #             = 2*sum_all T^2 - 2*sum_active T - sum_active T^2 + 112
        # This is not quite right because the inactive part is just T^2.
        
        # Let's just compute:
        # err_active = sum((T[active] - 1)^2)
        # err_inactive = sum(T[inactive]^2)
        
        # Vectorized active error:
        T_active = np.array([T[a,b,c] for a,b,c in active_indices])
        err_active = np.sum((T_active - 1.0)**2)
        
        # Vectorized inactive error:
        # Total sum of squares of T
        total_ss = np.sum(T**2)
        active_ss = np.sum(T_active**2)
        err_inactive = total_ss - active_ss
        
        return err_active + err_inactive

    def verify_fast(U, V, W, atol=1e-6):
        """
        Strict verification.
        """
        T = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        
        # Check active
        for a, b, c in active_indices:
            if abs(T[a, b, c] - 1.0) > atol:
                return False
        
        # Check all inactive
        # We can do this by checking if any inactive entry is non-zero.
        # Since T is small, we can just iterate or use a mask.
        # Let's create a mask once outside? No, T changes.
        # Let's just check the max absolute value of inactive entries.
        
        # Create a boolean mask for active indices
        # T shape (16, 28, 28)
        mask = np.zeros((16, 28, 28), dtype=bool)
        for a, b, c in active_indices:
            mask[a, b, c] = True
            
        inactive_vals = T[~mask]
        if np.any(np.abs(inactive_vals) > atol):
            return False
            
        return True

    def snap_to_halfint(U, V, W, threshold=1e-5):
        """Snap values to nearest half-integer."""
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        U_s[np.abs(U_s) < threshold] = 0
        V_s[np.abs(V_s) < threshold] = 0
        W_s[np.abs(W_s) < threshold] = 0
        return U_s, V_s, W_s

    def try_find_decomposition(target_rank, rng, max_time_per_trial=10.0):
        """
        Attempt to find a decomposition of rank target_rank.
        Returns (U, V, W) if successful, else None.
        """
        start_trial_time = time.time()
        
        # Initial guess: Random or structured?
        # Random is better for escaping local minima.
        # We want sparse solutions. 
        # L-BFGS-B minimizes f(x). 
        # f(x) = Error + lambda * L1_penalty
        
        # To encourage sparsity, we can add a penalty term.
        # However, L1 is not differentiable. 
        # We can use a smooth L1 approximation: x * log(1 + x^2/eps) or similar.
        # Or just rely on the fact that for small ranks, the solution might naturally be sparse if we start sparse?
        # Actually, random start is dense.
        
        # Let's try a different approach: 
        # Start with a dense random matrix, optimize to minimize Error.
        # Then, post-process to snap to half-integers and verify.
        # If it fails, try again with different seeds.
        
        # Parameters:
        # x = concat(U.flatten(), V.flatten(), W.flatten())
        
        dim_u = n * m * target_rank
        dim_v = m * p * target_rank
        dim_w = p * n * target_rank
        total_dim = dim_u + dim_v + dim_w
        
        # We need to be careful with the number of parameters.
        # For rank 97: 16*97 + 28*97 + 28*97 = 1552 + 2716 + 2716 = 6984 parameters.
        # L-BFGS-B on 7000 parameters is feasible but might be slow if many iterations are needed.
        
        def objective(x):
            if time.time() > start_trial_time + max_time_per_trial:
                return 1e10 # Timeout penalty
            
            U_flat = x[:dim_u]
            V_flat = x[dim_u:dim_u+dim_v]
            W_flat = x[dim_u+dim_v:]
            
            U = U_flat.reshape((n*m, target_rank))
            V = V_flat.reshape((m*p, target_rank))
            W = W_flat.reshape((p*n, target_rank))
            
            err = compute_error(U, V, W)
            
            # Add a small L2 regularization to prevent divergence?
            # Or L1 to encourage sparsity?
            # Let's add a small L1 penalty to push towards sparsity.
            # L1 = sum(|U| + |V| + |W|)
            # This is non-smooth, but L-BFGS-B might handle it okay if we use a smooth approximation?
            # Actually, L-BFGS-B assumes smoothness. 
            # Let's use a smooth L1: sum(2*sqrt(x^2+eps) - 2*sqrt(eps))
            # Or just rely on the error term. If the error is 0, the solution is valid.
            # We don't strictly need sparsity for the score, only for the "integer/half-integer" requirement.
            # So we can just find *any* real solution, then snap it.
            
            return err
        
        # We will run multiple random restarts
        best_solution = None
        best_err = np.inf
        
        num_restarts = 5
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            # Scale: We want the product U*V*W to be around 1.
            # If U,V,W are random normal, the sum of R terms will be large.
            # Let's scale by 1/sqrt(R) or similar.
            scale = 1.0 / np.sqrt(target_rank)
            
            U0 = rng.standard_normal((n*m, target_rank)) * scale
            V0 = rng.standard_normal((m*p, target_rank)) * scale
            W0 = rng.standard_normal((p*n, target_rank)) * scale
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            # Optimize
            res = opt.minimize(objective, x0, method='L-BFGS-B', 
                               options={'maxiter': 200, 'maxfun': 1000, 'disp': False})
            
            if res.success or res.fun < best_err:
                best_err = res.fun
                if best_err < 1e-4: # Good enough to try snapping
                    U_opt = res.x[:dim_u].reshape((n*m, target_rank))
                    V_opt = res.x[dim_u:dim_u+dim_v].reshape((m*p, target_rank))
                    W_opt = res.x[dim_u+dim_v:].reshape((p*n, target_rank))
                    
                    # Try snapping
                    Us, Vs, Ws = snap_to_halfint(U_opt, V_opt, W_opt)
                    if verify_fast(Us, Vs, Ws):
                        return Us, Vs, Ws
                    
                    # Try verifying without snapping (in case it's exact real)
                    if verify_fast(U_opt, V_opt, W_opt, atol=1e-5):
                        # If it's exact real, we might still be able to snap it?
                        # The problem requires integer/half-integer entries.
                        # If it's not close to half-integers, we can't use it directly.
                        # But maybe a different scaling or shift?
                        # For now, return None if snapping fails.
                        pass
                    
            # Check time
            if time.time() > start_trial_time + max_time_per_trial:
                break

        return None

    # Strategy:
    # 1. Try to find a rank 97 solution using optimization.
    # 2. If found, return it.
    # 3. If not, try rank 96, 95, etc.
    # 4. If all fail, fallback to the known Strassen+Schoolbook (rank 97) or Schoolbook (112).
    
    rng = np.random.default_rng(42)
    
    # First, let's have the fallback ready
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[a_idx(i, j), r] = 1
                    V[b_idx(j, k), r] = 1
                    W[c_idx(k, i), r] = 1
                    r += 1
        return U, V, W

    def strassen_4x4x4_plus_schoolbook_4x4x3():
        """Construct rank-97 decomposition: Strassen for 4x4x4 + schoolbook for 4x4x3."""
        rank = 0
        U_list = []
        V_list = []
        W_list = []

        def add_term(u_entries, v_entries, w_entries):
            nonlocal rank
            r = rank
            rank += 1
            for (idx, val) in u_entries:
                U_list.append((idx, r, val))
            for (idx, val) in v_entries:
                V_list.append((idx, r, val))
            for (idx, val) in w_entries:
                W_list.append((idx, r, val))

        # Strassen 4x4x4: 49 terms
        # This is the same implementation as in the seed program, which is correct.
        # I will reuse the logic from the seed program here to ensure correctness of the fallback.
        
        products = [
            {
                'U': [((0,0), 1), ((1,1), 1)],
                'V': [((0,0), 1), ((1,1), 1)],
                'W': [((0,0), 1), ((1,1), 1)],
            },
            {
                'U': [((1,0), 1), ((1,1), 1)],
                'V': [((0,0), 1)],
                'W': [((0,1), 1)],
            },
            {
                'U': [((0,0), 1)],
                'V': [((0,1), 1), ((1,1), -1)],
                'W': [((1,0), 1)],
            },
            {
                'U': [((1,1), 1)],
                'V': [((1,0), 1), ((0,0), -1)],
                'W': [((0,0), 1), ((1,1), 1)],
            },
            {
                'U': [((0,0), 1), ((0,1), 1)],
                'V': [((1,1), 1)],
                'W': [((0,0), 1), ((1,0), 1)],
            },
            {
                'U': [((1,0), 1), ((0,0), -1)],
                'V': [((0,0), 1), ((0,1), 1)],
                'W': [((1,1), 1)],
            },
            {
                'U': [((0,1), 1), ((1,1), -1)],
                'V': [((1,0), 1), ((1,1), 1)],
                'W': [((0,0), 1), ((0,1), 1)],
            },
        ]

        strassen_terms = [
            ([((0,0), 1), ((1,1), 1)],
             [((0,0), 1), ((1,1), 1)],
             [((0,0), 1), ((1,1), 1)]),
            ([((1,0), 1), ((1,1), 1)],
             [((0,0), 1)],
             [((1,0), 1)]),
            ([((0,0), 1)],
             [((0,1), 1), ((1,1), -1)],
             [((0,1), 1)]),
            ([((1,1), 1)],
             [((1,0), 1), ((0,0), -1)],
             [((0,0), 1), ((1,1), 1)]),
            ([((0,0), 1), ((0,1), 1)],
             [((1,1), 1)],
             [((0,0), 1), ((0,1), 1)]),
            ([((1,0), 1), ((0,0), -1)],
             [((0,0), 1), ((0,1), 1)],
             [((1,1), 1)]),
            ([((0,1), 1), ((1,1), -1)],
             [((1,0), 1), ((1,1), 1)],
             [((0,0), 1), ((1,0), 1)]),
        ]

        for prod in products:
            U_blocks = prod['U']
            V_blocks = prod['V']
            W_blocks = prod['W']

            VA = {}
            for (bi, bj), coeff in U_blocks:
                for di in range(2):
                    for dj in range(2):
                        gi = 2 * bi + di
                        gj = 2 * bj + dj
                        gidx = a_idx(gi, gj)
                        key = (di, dj)
                        if key not in VA:
                            VA[key] = {}
                        VA[key][gidx] = VA[key].get(gidx, 0) + coeff

            VB = {}
            for (bj, bk), coeff in V_blocks:
                for dj in range(2):
                    for dk in range(2):
                        gj = 2 * bj + dj
                        gk = 2 * bk + dk
                        gidx = b_idx(gj, gk)
                        key = (dj, dk)
                        if key not in VB:
                            VB[key] = {}
                        VB[key][gidx] = VB[key].get(gidx, 0) + coeff

            WC = {}
            for (bk, bi), coeff in W_blocks:
                for dk in range(2):
                    for di in range(2):
                        gk = 2 * bk + dk
                        gi = 2 * bi + di
                        gidx = c_idx(gk, gi)
                        key = (dk, di)
                        if key not in WC:
                            WC[key] = {}
                        WC[key][gidx] = WC[key].get(gidx, 0) + coeff

            for (u_ents, v_ents, w_ents) in strassen_terms:
                u_global = []
                for (key, sc) in u_ents:
                    if key in VA:
                        for gidx, vc in VA[key].items():
                            u_global.append((gidx, sc * vc))

                v_global = []
                for (key, sc) in v_ents:
                    if key in VB:
                        for gidx, vc in VB[key].items():
                            v_global.append((gidx, sc * vc))

                w_global = []
                for (key, sc) in w_ents:
                    if key in WC:
                        for gidx, vc in WC[key].items():
                            w_global.append((gidx, sc * vc))

                u_global = [(idx, c) for idx, c in u_global if c != 0]
                v_global = [(idx, c) for idx, c in v_global if c != 0]
                w_global = [(idx, c) for idx, c in w_global if c != 0]

                if u_global and v_global and w_global:
                    add_term(u_global, v_global, w_global)

        # Schoolbook for 4x4x3 (k=4,5,6)
        for i in range(n):
            for j in range(m):
                for k in range(4, p):
                    add_term(
                        [(a_idx(i, j), 1)],
                        [(b_idx(j, k), 1)],
                        [(c_idx(k, i), 1)]
                    )

        r = rank
        U_mat = np.zeros((n * m, r))
        V_mat = np.zeros((m * p, r))
        W_mat = np.zeros((p * n, r))
        for (row, rr, val) in U_list:
            U_mat[row, rr] = val
        for (row, rr, val) in V_list:
            V_mat[row, rr] = val
        for (row, rr, val) in W_list:
            W_mat[row, rr] = val

        return U_mat, V_mat, W_mat

    # 1. Get the fallback solution (Rank 97)
    U_fallback, V_fallback, W_fallback = strassen_4x4x4_plus_schoolbook_4x4x3()
    best_U, best_V, best_W = U_fallback, V_fallback, W_fallback
    best_rank = U_fallback.shape[1]
    
    # Verify fallback
    if not verify_fast(best_U, best_V, best_W):
        # Should not happen, but just in case
        U_fallback, V_fallback, W_fallback = schoolbook()
        best_U, best_V, best_W = U_fallback, V_fallback, W_fallback
        best_rank = U_fallback.shape[1]

    # 2. Try to find better solutions using optimization
    # We start from best_rank - 1 and go down.
    # We limit the time spent per rank to ensure we finish.
    
    # How much time do we have?
    time_elapsed = time.time() - start_time
    time_remaining = deadline - time_elapsed
    
    # Allocate time to search for lower ranks
    # Let's try ranks: 96, 95, 94, 93, 92, 91, 90
    # We can spend about 10-15 seconds per rank if we have time.
    
    search_ranks = list(range(best_rank - 1, best_rank - 10, -1))
    
    # If we have plenty of time, we can try more.
    # But L-BFGS-B on ~7000 vars might be slow.
    
    for target_rank in search_ranks:
        if time.time() > deadline - 10:
            break
            
        # Try to find a solution of this rank
        result = try_find_decomposition(target_rank, rng, max_time_per_trial=15.0)
        
        if result is not None:
            U_new, V_new, W_new = result
            # Final verification
            if verify_fast(U_new, V_new, W_new):
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = target_rank
                # If we found a lower rank, we can try even lower, but we should also
                # consider if we have time to continue.
                # The loop continues to the next lower rank.
                continue
            else:
                # Verification failed, discard
                pass
        
        # If we didn't find a solution, we can still continue to the next rank
        # but we might be running out of time.

    # Final check
    if not verify_fast(best_U, best_V, best_W):
        # Last resort
        best_U, best_V, best_W = schoolbook()
        best_rank = n * m * p

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
