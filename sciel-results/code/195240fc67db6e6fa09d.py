import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin for 120s limit
    
    # Target tensor M[alpha, beta, gamma]
    # alpha: index in W (k*n + i)
    # beta: index in U (i*m + j)
    # gamma: index in V (j*p + k)
    # M[alpha, beta, gamma] = 1 if (i,j,k) valid, else 0
    
    # We represent the current "residual" tensor R which we want to decompose.
    # Initially R = M.
    # We maintain U, V, W as lists of columns (each column is a vector).
    # Actually, we build U, V, W as numpy arrays of shape (dim, max_rank) and fill columns.
    
    # Max possible rank is n*m*p = 60.
    max_rank = n * m * p
    U = np.zeros((n*m, max_rank))
    V = np.zeros((m*p, max_rank))
    W = np.zeros((p*n, max_rank))
    
    # Residual tensor
    # To save memory and time, we can just track which entries are still 1.
    # Total entries 20*6*30 = 3600. It's small enough to keep as a dense array if needed,
    # but a set of tuples (alpha, beta, gamma) for non-zeros is faster for iteration.
    remaining = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = k * n + i
                beta = i * m + j
                gamma = j * p + k
                remaining.add((alpha, beta, gamma))
    
    rank = 0
    best_rank = max_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Greedy Strategy:
    # At each step, pick a remaining entry (a, b, g).
    # Set U[b, rank] = 1, V[g, rank] = 1, W[a, rank] = 1.
    # This satisfies M[a,b,g] = 1 * 1 * 1 = 1.
    # However, this adds 1 to ALL other entries (a', b, g), (a, b', g), (a, b, g') that share indices.
    # Specifically, the term added is u_b * v_g * w_a.
    # If we set them all to 1, we add 1 to:
    # - All (a', b, g) for a' in [0, pn)
    # - All (a, b', g) for b' in [0, nm)
    # - All (a, b, g') for g' in [0, mp)
    # Note: overlaps are counted once in the tensor addition, but logically we are adding a rank-1 tensor.
    # The new residual R_new = R_old - u_b v_g w_a.
    
    # Since we start with R having 1s at specific spots and 0s elsewhere,
    # and we subtract 1s, some entries might become -1, 2, etc.
    # The goal is to reach R = 0.
    
    # To be smarter than pure greedy (which is schoolbook if we pick disjoint tuples),
    # we can look for a tuple that covers the most "complex" part?
    # Actually, standard greedy for tensor rank minimization on 0/1 tensors often just picks any available.
    # But we can use a heuristic: pick the entry (a,b,g) that minimizes the number of *new* non-zeros created in the residual?
    # Or simply: pick the entry with the highest "degree" (number of other non-zrows sharing indices)?
    
    # Let's try a simple heuristic: Pick the remaining entry (a,b,g) such that the number of
    # remaining entries sharing a, b, or g is maximized. This "clusters" the solution.
    
    while remaining and time.time() < deadline:
        if rank >= max_rank:
            break
            
        # Heuristic: Calculate score for each candidate (a,b,g) in remaining
        # Score = count of other entries in `remaining` that share index a OR b OR g.
        # This is O(R * N) where N is number of remaining entries.
        # N starts at 60, decreases. R is up to 60. Very fast.
        
        best_score = -1
        best_candidate = None
        
        # Pre-compute counts for each index to speed up?
        # count_a[a] = number of remaining entries with alpha=a
        # count_b[b] = number of remaining entries with beta=b
        # count_g[g] = number of remaining entries with gamma=g
        # For a candidate (a,b,g), the number of entries affected (including itself) is:
        # count_a[a] + count_b[b] + count_g[g] - (overlaps)
        # Overlaps are tricky. But as a proxy, sum of counts is good.
        
        count_a = np.zeros(p*n, dtype=int)
        count_b = np.zeros(n*m, dtype=int)
        count_g = np.zeros(m*p, dtype=int)
        
        for a, b, g in remaining:
            count_a[a] += 1
            count_b[b] += 1
            count_g[g] += 1
            
        for a, b, g in remaining:
            score = count_a[a] + count_b[b] + count_g[g]
            if score > best_score:
                best_score = score
                best_candidate = (a, b, g)
                
        if best_candidate is None:
            break
            
        a, b, g = best_candidate
        
        # Commit this rank-1 term
        U[b, rank] = 1.0
        V[g, rank] = 1.0
        W[a, rank] = 1.0
        
        # Update residual
        # R_new[x, y, z] = R_old[x, y, z] - (1 if y==b and z==g and x==a ... wait)
        # The term is U[b, r] * V[g, r] * W[a, r].
        # Since U[b,r]=1, V[g,r]=1, W[a,r]=1, the term is 1 at (a,b,g) and 0 elsewhere?
        # NO. The tensor product is:
        # T_r[alpha, beta, gamma] = W[alpha, r] * U[beta, r] * V[gamma, r]
        # Since W is zero except at a, U zero except at b, V zero except at g:
        # T_r is 1 ONLY at (a, b, g) and 0 everywhere else.
        
        # Wait, if U, V, W are sparse one-hots, then the rank-1 tensor is a single point.
        # This is just the schoolbook decomposition!
        # To get a lower rank, we MUST use non-sparse U, V, W.
        # The greedy approach above with 1s and 0s only yields rank 60.
        
        # STRATEGY CORRECTION:
        # We need to allow U, V, W to have multiple non-zeros per column.
        # Let's try a different structural approach:
        # Fix the structure of U and V to be "blocky" or use Strassen-like patterns?
        # Or, use a solver for small ranks?
        
        # Given the time, let's try a specific algebraic construction for small matrices.
        # For 2x3x10, is there a known lower bound?
        # Lower bound for matrix multiplication M_{n,m,p} is max(nm, mp, pn).
        # Here max(6, 30, 20) = 30.
        # So rank >= 30.
        # Schoolbook is 60.
        # Can we achieve 30?
        
        # Let's try to construct a decomposition of rank 30.
        # If we can find one, it's optimal.
        
        # Strategy: Try to find a decomposition of rank R=30.
        # We can use a randomized search with local improvement, but constrained to integer/half-integer values?
        # Or, we can use the fact that for rectangular matrices, sometimes we can "compress" the dimension.
        
        # Let's try a specific pattern.
        # Consider the output C = AB.
        # C is 2x10.
        # A is 2x3, B is 3x10.
        
        # If we use the "middle" dimension m=3.
        # Rank 30 is the lower bound.
        
        # Let's attempt a systematic search for rank 30 using a simplified model.
        # If we fix the first 20 columns of W to correspond to the 20 outputs?
        
        # Actually, let's just return the schoolbook decomposition if we can't find better quickly.
        # But the prompt asks for a DIFFERENT strategy.
        # The previous strategy was random guessing.
        # This strategy is: **Algebraic Construction for Lower Bound**.
        
        # Let's try to build rank 30.
        # We need U (6x30), V (30x30), W (20x30).
        
        # Hypothesis: We can pair the 3 intermediate steps with the 10 columns?
        
        # Let's try a randomized search for R=30 with a specific initialization:
        # Initialize U, V, W with random integers in {-1, 0, 1}.
        # Use gradient descent? No, non-differentiable.
        # Use a local search that flips bits?
        
        # Let's try a simpler heuristic:
        # We know rank >= 30.
        # Let's try to construct a valid rank 30 decomposition by solving a system of equations for specific structures.
        
        # If we assume U has rows corresponding to (i,j).
        # Let's try to make V an identity-like matrix? No, V is 30x30.
        
        # Alternative: Use the known result that for M_{2,3,10}, the rank is likely 30 or slightly above.
        # Let's try a random restart local search for R=30.
        
        pass 

    # The greedy loop above with 1-hots is useless for rank reduction.
    # Let's implement a proper local search for R=30.
    
    def try_local_search(R, num_trials=50, steps_per_trial=1000):
        start_time = time.time()
        for t in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Initialize with small integers
            U = np.random.choice([-1, 0, 1], size=(n*m, R))
            V = np.random.choice([-1, 0, 1], size=(m*p, R))
            W = np.random.choice([-1, 0, 1], size=(p*n, R))
            
            # Ensure no column is all zeros (waste)
            for c in range(R):
                if np.all(U[:, c] == 0) or np.all(V[:, c] == 0) or np.all(W[:, c] == 0):
                    U[np.random.randint(n*m), c] = 1
                    V[np.random.randint(m*p), c] = 1
                    W[np.random.randint(p*n), c] = 1
                    
            # Local Search
            # Objective: Minimize sum of squared errors on all tensor entries.
            # But we only care about exact match.
            # Let's check the error.
            
            def get_error():
                # Compute T = W @ (U * V) ? No.
                # T[a,b,g] = sum_c W[a,c] U[b,c] V[g,c]
                # We can compute this efficiently.
                # T = np.einsum('ac,bc,gc->abg', W, U, V)
                # This is 20*6*30 = 3600 entries.
                # 3600 * 30 = 108,000 ops. Very fast.
                T = np.einsum('ac,bc,gc->abg', W, U, V)
                
                # Target M
                # M is 1 at (k*n+i, i*m+j, j*p+k)
                # We can construct M once outside.
                # Let's assume M is available globally or built here.
                
                err = np.sum((T - M_target)**2)
                return err, T

            M_target = np.zeros((p*n, n*m, m*p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        M_target[k*n+i, i*m+j, j*p+k] = 1.0
                        
            err, T = get_error()
            
            # If error is 0, we found it!
            if err < 1e-9:
                return U, V, W
                
            # Local Search Step
            # Pick a random matrix (U, V, or W), a random row, a random column.
            # Change the value by +/- 1 or 0.
            # If error decreases, accept.
            
            for step in range(steps_per_trial):
                if time.time() > deadline:
                    return None
                    
                # Pick random change
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    mat = U
                    dim = n*m
                elif mat_idx == 1:
                    mat = V
                    dim = m*p
                else:
                    mat = W
                    dim = p*n
                    
                row = np.random.randint(dim)
                col = np.random.randint(R)
                
                old_val = mat[row, col]
                # Try new values: old_val + delta
                # Deltas: -1, 0, 1
                # Also try 0 if not 0
                candidates = [old_val - 1, old_val, old_val + 1]
                if 0 not in candidates:
                    candidates.append(0)
                    
                best_new_val = old_val
                best_err = err
                
                for new_val in candidates:
                    if new_val == old_val:
                        continue
                    # Apply change
                    mat[row, col] = new_val
                    new_err, _ = get_error()
                    if new_err < best_err:
                        best_err = new_err
                        best_new_val = new_val
                    # Revert
                    mat[row, col] = old_val
                    
                # Apply best change
                if best_new_val != old_val:
                    mat[row, col] = best_new_val
                    err = best_err
                    
                if err < 1e-9:
                    return U, V, W
                    
        return None

    # Try ranks from lower bound 30 upwards
    for R in range(30, 40):
        if time.time() > deadline - 5:
            break
            
        res = try_local_search(R, num_trials=20, steps_per_trial=2000)
        if res is not None:
            U, V, W = res
            # Verify strictly
            T = np.einsum('ac,bc,gc->abg', W, U, V)
            M_target = np.zeros((p*n, n*m, m*p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        M_target[k*n+i, i*m+j, j*p+k] = 1.0
            if np.allclose(T, M_target, atol=1e-9):
                return {"U": U, "V": V, "W": W}
                
    # Fallback to schoolbook
    rank_seed = n * m * p
    U_seed = np.zeros((n*m, rank_seed))
    V_seed = np.zeros((m*p, rank_seed))
    W_seed = np.zeros((p*n, rank_seed))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j][column] = 1
                V_seed[j * p + k][column] = 1
                W_seed[k * n + i][column] = 1
                column += 1
                
    return {"U": U_seed, "V": V_seed, "W": W_seed}
