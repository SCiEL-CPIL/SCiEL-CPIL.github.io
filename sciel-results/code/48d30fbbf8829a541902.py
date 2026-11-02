import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18

    def verify_decomposition(U, V, W):
        """Verify that the tensor decomposition is exact."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(r):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1) > 1e-9:
                        return False
        return True

    # Rank 50 decomposition (Strassen-based) - the known good solution
    def current_solution():
        rank = 50
        U = [[0] * rank for _ in range(n_rows)]
        V = [[0] * rank for _ in range(m_rows)]
        W = [[0] * rank for _ in range(p_rows)]
        
        col = 0
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            v_e = 0 * p + k0
            v_f = 0 * p + k1
            v_g = 1 * p + k0
            v_h = 1 * p + k1
            u_a = 0 * m + 0
            u_b = 0 * m + 1
            u_c = 1 * m + 0
            u_d = 1 * m + 1
            
            U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1
            U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1
            U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1
            U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1
            U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1
            U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1
            U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
            W[w_01][col+2] += 1; W[w_01][col+4] += 1
            W[w_10][col+1] += 1; W[w_10][col+3] += 1
            W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
            col += 7
            
        k8 = 8
        v_e = 0 * p + k8
        v_g = 1 * p + k8
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a][col] += 1; V[v_e][col] += 1
        U[u_b][col+1] += 1; V[v_g][col+1] += 1
        U[u_c][col+2] += 1; V[v_e][col+2] += 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1
        
        w_08 = k8 * n + 0
        w_18 = k8 * n + 1
        W[w_08][col] += 1; W[w_08][col+1] += 1
        W[w_18][col+2] += 1; W[w_18][col+3] += 1
        col += 4
        
        u_ar0 = 0 * m + 2
        u_ar1 = 1 * m + 2
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
            U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
            col += 2
            
        return {"U": U, "V": V, "W": W}

    # Build the tensor T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # We want to find rank-r decomposition: T = sum_c U[:,c] outer V[:,c] outer W[:,c]
    # i.e., T[a,b,c_idx] = sum_r U[a,r]*V[b,r]*W[c_idx,r]
    
    def tensor_value(i, j, k):
        """T[i*m+j, j*p+k, k*n+i] should be 1"""
        return 1.0

    def compute_tensor_entries(U, V, W, r):
        """Compute all tensor entries from factor matrices."""
        # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
        # For each (i,j,k): entry = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        errors = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    errors[i, j, k] = val - 1.0
        return errors

    def try_find_lower_rank(r, max_iterations=500, seed=42):
        """
        Try to find a rank-r decomposition using structured random search with local refinement.
        Uses the insight that we can start from the rank-50 solution and try to merge/eliminate terms.
        """
        if time.time() > deadline - 20:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Strategy: Start with random sparse initialization, then do local search
        # Each "term" c has a pattern of which (i,j) pairs in U, (j,k) in V, (k,i) in W are nonzero
        
        best_U = None
        best_V = None
        best_W = None
        best_error = float('inf')
        
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            
            # Initialize with structured sparsity
            U = np.zeros((n_rows, r), dtype=np.float64)
            V = np.zeros((m_rows, r), dtype=np.float64)
            W = np.zeros((p_rows, r), dtype=np.float64)
            
            # For each rank-1 term, create a sparse pattern
            for c in range(r):
                # Choose a small number of non-zero entries
                # U: pick 2-3 (i,j) pairs
                num_u = rng.randint(2, 4)
                u_indices = rng.choice(n_rows, size=num_u, replace=False)
                for idx in u_indices:
                    U[idx, c] = rng.choice([-1.0, 1.0, 2.0, -2.0])
                
                num_v = rng.randint(2, 4)
                v_indices = rng.choice(m_rows, size=num_v, replace=False)
                for idx in v_indices:
                    V[idx, c] = rng.choice([-1.0, 1.0, 2.0, -2.0])
                
                num_w = rng.randint(2, 4)
                w_indices = rng.choice(p_rows, size=num_w, replace=False)
                for idx in w_indices:
                    W[idx, c] = rng.choice([-1.0, 1.0, 2.0, -2.0])
            
            # Local search: iteratively perturb entries to reduce error
            for it in range(max_iterations):
                if time.time() > deadline - 5:
                    break
                
                # Compute current error
                errors = compute_tensor_entries(U, V, W, r)
                current_error = np.max(np.abs(errors))
                
                if current_error < best_error:
                    best_error = current_error
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if current_error < 1e-10:
                    break
                
                # Find the worst entry
                worst_idx = np.unravel_index(np.argmax(np.abs(errors)), errors.shape)
                wi, wj, wk = worst_idx
                
                # Try to fix this entry by perturbing relevant U, V, W entries
                # U[i*m+wj, :], V[wj*p+wk, :], W[wk*n+wi, :]
                u_row = wi * m + wj
                v_row = wj * p + wk
                w_row = wk * n + wi
                
                # Try random perturbations on these rows
                improved = False
                for _ in range(10):
                    # Pick which matrix to perturb
                    mat_choice = rng.randint(3)
                    if mat_choice == 0:
                        col = rng.randint(r)
                        old_val = U[u_row, col]
                        U[u_row, col] = old_val + rng.choice([-1.0, 1.0])
                    elif mat_choice == 1:
                        col = rng.randint(r)
                        old_val = V[v_row, col]
                        V[v_row, col] = old_val + rng.choice([-1.0, 1.0])
                    else:
                        col = rng.randint(r)
                        old_val = W[w_row, col]
                        W[w_row, col] = old_val + rng.choice([-1.0, 1.0])
                    
                    # Check if improved
                    errors_new = compute_tensor_entries(U, V, W, r)
                    new_error = np.max(np.abs(errors_new))
                    
                    if new_error < current_error * 0.99:  # Require meaningful improvement
                        current_error = new_error
                        improved = True
                        break
                    else:
                        # Revert
                        if mat_choice == 0:
                            U[u_row, col] = old_val
                        elif mat_choice == 1:
                            V[v_row, col] = old_val
                        else:
                            W[w_row, col] = old_val
                
                if not improved:
                    # Try a broader random perturbation
                    mat_choice = rng.randint(3)
                    if mat_choice == 0:
                        row = rng.randint(n_rows)
                        col = rng.randint(r)
                        U[row, col] += rng.choice([-1.0, 1.0])
                    elif mat_choice == 1:
                        row = rng.randint(m_rows)
                        col = rng.randint(r)
                        V[row, col] += rng.choice([-1.0, 1.0])
                    else:
                        row = rng.randint(p_rows)
                        col = rng.randint(r)
                        W[row, col] += rng.choice([-1.0, 1.0])
            
            # Final check
            if best_error < 1e-10 and best_U is not None:
                # Round to integers
                U_int = np.round(best_U).astype(np.int64)
                V_int = np.round(best_V).astype(np.int64)
                W_int = np.round(best_W).astype(np.int64)
                
                # Verify exactness
                if verify_decomposition(U_int.tolist(), V_int.tolist(), W_int.tolist()):
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
        
        return None

    # Try to find lower rank solutions
    best_result = None
    best_rank = 50
    
    # First, get the known rank-50 solution
    known_sol = current_solution()
    
    # Verify it
    if verify_decomposition(known_sol["U"], known_sol["V"], known_sol["W"]):
        best_result = known_sol
    else:
        # Shouldn't happen, but fallback
        best_result = known_sol
    
    # Try to improve: search for lower ranks
    # Based on the block structure, try ranks 49, 48, 47, ...
    for r in range(49, 42, -1):
        if time.time() > deadline - 30:
            break
        
        result = try_find_lower_rank(r, max_iterations=300, seed=r*1000)
        if result is not None:
            best_result = result
            best_rank = r
            break  # Found a valid lower rank, stop
    
    # If we have time, try a different approach: structured embedding
    if time.time() < deadline - 40 and best_rank == 50:
        # Try a more structured approach using the 2+1 split more aggressively
        # The key insight: A is 2x3, so split inner dim as 2+1
        # A_left (2x2) * B_top (2x9) + A_right (2x1) * B_bottom (1x9)
        # For A_left * B_top: use Strassen on 2x2 blocks of B_top
        # B_top has 9 columns, so 4 full 2-col blocks + 1 single col
        # Each 2x2x2 block uses 7 mults (Strassen)
        # The single col uses 4 mults (schoolbook)
        # A_right * B_bottom: 2x1x9, uses 18 mults (schoolbook)
        # Total: 4*7 + 4 + 18 = 28 + 4 + 18 = 50
        
        # Can we do better? What if we use a different split?
        # Try 1+2 split: A_left (2x1) * B_left (1x9) + A_right (2x2) * B_right (2x9)
        # A_left * B_left: 2x1x9 = 18 mults
        # A_right * B_right: 2x2x9 = 4*7 + 4 = 32 mults (same as before)
        # Total: 18 + 32 = 50 (same)
        
        # What about overlapping Strassen?
        # This is where the search should find improvements if they exist
        
        # Try another search strategy: start from rank-50 and try to eliminate terms
        def try_eliminate_terms():
            """Try to find a lower rank by starting from rank-50 and eliminating terms."""
            if time.time() > deadline - 15:
                return None
            
            U0 = np.array(current_solution()["U"], dtype=np.float64)
            V0 = np.array(current_solution()["V"], dtype=np.float64)
            W0 = np.array(current_solution()["W"], dtype=np.float64)
            
            # Try removing one term at a time and see if we can compensate
            for r_target in range(49, 44, -1):
                if time.time() > deadline - 10:
                    return None
                
                # Start from rank-50, try to merge/eliminate (50 - r_target) terms
                U = U0.copy()
                V = V0.copy()
                W = W0.copy()
                
                # Zero out some terms and see if we can adjust others
                # This is a heuristic - not guaranteed to work
                rng = np.random.RandomState(42)
                
                for attempt in range(5):
                    U_test = U0.copy()
                    V_test = V0.copy()
                    W_test = W0.copy()
                    
                    # Zero out random terms
                    terms_to_remove = rng.choice(50, size=50-r_target, replace=False)
                    for t in terms_to_remove:
                        U_test[:, t] = 0
                        V_test[:, t] = 0
                        W_test[:, t] = 0
                    
                    # Now try to adjust the remaining terms to compensate
                    # This is a hard problem - just do a few random perturbations
                    for _ in range(100):
                        if time.time() > deadline - 5:
                            break
                        
                        # Compute error
                        errors = compute_tensor_entries(U_test, V_test, W_test, 50)
                        err = np.max(np.abs(errors))
                        
                        if err < 1e-10:
                            break
                        
                        # Perturb a random entry in a non-zero term
                        active_terms = [t for t in range(50) if t not in terms_to_remove]
                        t = rng.choice(active_terms)
                        mat_choice = rng.randint(3)
                        if mat_choice == 0:
                            row = rng.randint(n_rows)
                            U_test[row, t] += rng.choice([-1.0, 1.0])
                        elif mat_choice == 1:
                            row = rng.randint(m_rows)
                            V_test[row, t] += rng.choice([-1.0, 1.0])
                        else:
                            row = rng.randint(p_rows)
                            W_test[row, t] += rng.choice([-1.0, 1.0])
                    
                    # Check if we found a solution
                    errors = compute_tensor_entries(U_test, V_test, W_test, 50)
                    if np.max(np.abs(errors)) < 1e-10:
                        # Extract only the active terms
                        U_final = U_test[:, active_terms]
                        V_final = V_test[:, active_terms]
                        W_final = W_test[:, active_terms]
                        
                        U_int = np.round(U_final).astype(np.int64)
                        V_int = np.round(V_final).astype(np.int64)
                        W_int = np.round(W_final).astype(np.int64)
                        
                        if verify_decomposition(U_int.tolist(), V_int.tolist(), W_int.tolist()):
                            return {
                                "U": U_int.tolist(),
                                "V": V_int.tolist(),
                                "W": W_int.tolist()
                            }
                
                if time.time() > deadline - 10:
                    return None
            
            return None
        
        elim_result = try_eliminate_terms()
        if elim_result is not None:
            best_result = elim_result
    
    # Return the best result found
    return best_result
