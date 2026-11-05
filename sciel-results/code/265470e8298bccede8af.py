"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _tensor_from_matrices(U, V, W):
    """Compute the tensor from factor matrices."""
    n_a, rank = U.shape
    n_b = V.shape[0]
    n_c = W.shape[0]
    T = np.zeros((n_a, n_b, n_c))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return T

def _make_rank17():
    """Build the known rank-17 decomposition."""
    n, m, p = 2, 3, 3
    n_a, n_b, n_c = 6, 9, 6
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    terms = []
    
    # Part 1: Strassen for A1(2x2) x B1_left(2x2) -> C[0:2, 0:2]
    terms.append(([0, 4], [1, 1], [0, 4], [1, 1], [0, 3], [1, 1]))
    terms.append(([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]))
    terms.append(([0], [1], [1, 4], [1, -1], [2, 3], [1, 1]))
    terms.append(([4], [1], [3, 0], [1, -1], [0, 1], [1, 1]))
    terms.append(([0, 1], [1, 1], [4], [1], [0, 2], [-1, 1]))
    terms.append(([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]))
    terms.append(([1, 4], [1, -1], [3, 4], [1, 1], [0], [1]))
    
    # Part 2: Direct for A1(2x2) x B1_right(2x1) -> C[:, 2]
    terms.append(([0], [1], [2], [1], [4], [1]))
    terms.append(([1], [1], [5], [1], [4], [1]))
    terms.append(([3], [1], [2], [1], [5], [1]))
    terms.append(([4], [1], [5], [1], [5], [1]))
    
    # Part 3: Direct for A2(2x1) x B2(1x3) -> C[0:2, 0:3]
    terms.append(([2], [1], [6], [1], [0], [1]))
    terms.append(([2], [1], [7], [1], [2], [1]))
    terms.append(([2], [1], [8], [1], [4], [1]))
    terms.append(([5], [1], [6], [1], [1], [1]))
    terms.append(([5], [1], [7], [1], [3], [1]))
    terms.append(([5], [1], [8], [1], [5], [1]))
    
    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        for a, val in zip(u_idx, u_val):
            U[a, r] = val
        for b, val in zip(v_idx, v_val):
            V[b, r] = val
        for c, val in zip(w_idx, w_val):
            W[c, r] = val
    
    return U, V, W

def _try_structured_rank16(deadline):
    """
    Try to find rank-16 by imposing linear dependencies among correction terms.
    
    Strategy: The 17-term solution has 7 Strassen terms + 10 correction terms.
    The correction terms involve rows of U/V that overlap with Strassen terms.
    We try to find if one correction term can be absorbed by adjusting others.
    
    Key observation: Terms 7-10 (Part 2) use U rows {0,1,3,4} and V rows {2,5}.
    Terms 11-16 (Part 3) use U rows {2,5} and V rows {6,7,8}.
    Strassen terms use U rows {0,1,3,4} and V rows {0,1,3,4}.
    
    We try: for each candidate term to drop, solve for adjustments to remaining
    terms that maintain exact equality. Use structured perturbation focused on
    the correction block.
    """
    n, m, p = 2, 3, 3
    n_a, n_b, n_c = 6, 9, 6
    T = _build_tensor(n, m, p)
    
    U17, V17, W17 = _make_rank17()
    
    # Identify which terms are "correction" terms (indices 7-16)
    correction_indices = list(range(7, 17))
    strassen_indices = list(range(7))
    
    # For each candidate term to eliminate, try to find adjustments
    # to the remaining 16 terms that maintain exact tensor equality.
    
    # Approach: Fix the Strassen block (terms 0-6) and only adjust correction terms.
    # This gives us a 10-term correction block that must produce:
    #   T_correction = T - sum_{r=0}^{6} U[:,r] V[:,r] W[:,r]
    # We need to represent T_correction with 9 rank-1 terms instead of 10.
    
    # Compute the correction tensor
    T_strassen = _tensor_from_matrices(U17[:, :7], V17[:, :7], W17[:, :7])
    T_corr = T - T_strassen
    
    # The correction terms operate on:
    # U rows: {0,1,2,3,4,5} but specifically:
    #   Terms 7-10: U rows {0,1,3,4}, V rows {2,5}, W rows {4,5}
    #   Terms 11-16: U rows {2,5}, V rows {6,7,8}, W rows {0,1,2,3,4,5}
    
    # Try: can we represent T_corr with 9 terms?
    # T_corr has support only on specific (a,b,c) triples.
    
    # Let's try a more direct approach: for each pair of correction terms,
    # check if their sum can be represented as a single rank-1 term plus
    # adjustments to other terms.
    
    # Actually, let's try a systematic linear algebra approach:
    # Fix the Strassen terms. The correction block has 10 rank-1 terms.
    # We want to find 9 rank-1 terms that give the same tensor.
    
    # The correction tensor T_corr has at most 10 non-zero entries.
    # Let's find them.
    nonzero = np.argwhere(T_corr != 0)
    # Each nonzero entry (a,b,c) must be covered by at least one correction term.
    
    # For 9 terms to cover 10 entries, at least one term must cover 2 entries.
    # A rank-1 term U[:,r] V[:,r] W[:,r] covers all (a,b,c) where U[a,r]!=0, V[b,r]!=0, W[c,r]!=0.
    
    # Let's try: for each pair of nonzero entries (a1,b1,c1) and (a2,b2,c2),
    # check if there exists a rank-1 term that covers both, and if the remaining
    # 8 entries can be covered by 8 rank-1 terms.
    
    # This is getting complex. Let's try a simpler approach:
    # Use random restarts with structured initialization.
    
    rng = np.random.default_rng(12345)
    
    # Initialize 9 correction terms from the 10 existing ones,
    # dropping one and perturbing the rest.
    
    for drop_idx in correction_indices:
        if time.time() > deadline - 20:
            break
        
        # Keep all correction terms except drop_idx
        keep_corr = [i for i in correction_indices if i != drop_idx]
        
        # Start from the kept terms
        U_corr = U17[:, keep_corr].copy()  # shape (6, 9)
        V_corr = V17[:, keep_corr].copy()  # shape (9, 9)
        W_corr = W17[:, keep_corr].copy()  # shape (6, 9)
        
        # Full 16-term candidate: Strassen + adjusted correction
        U_try = np.hstack([U17[:, :7], U_corr])
        V_try = np.hstack([V17[:, :7], V_corr])
        W_try = np.hstack([W17[:, :7], W_corr])
        
        # Compute residual
        T_try = _tensor_from_matrices(U_try, V_try, W_try)
        residual = T - T_try
        
        # Now try to fix the residual by adjusting the 9 correction terms
        # Use iterative refinement: adjust one term at a time
        
        success = False
        for iteration in range(200):
            if time.time() > deadline - 15:
                break
            
            T_try = _tensor_from_matrices(U_try, V_try, W_try)
            residual = T - T_try
            max_err = np.max(np.abs(residual))
            
            if max_err < 1e-10:
                success = True
                break
            
            # Find the largest residual entry
            a_max, b_max, c_max = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
            err_val = residual[a_max, b_max, c_max]
            
            if abs(err_val) < 1e-12:
                break
            
            # Try to fix by adjusting one of the 9 correction terms
            # Pick the term that best covers this entry
            best_term = -1
            best_score = -1e18
            
            for t in range(9):
                u_val = U_try[a_max, 7 + t]
                v_val = V_try[b_max, 7 + t]
                w_val = W_try[c_max, 7 + t]
                # The term contributes u_val * v_val * w_val to (a_max, b_max, c_max)
                # We want to adjust to fix the error
                # If we scale the term, we can change the contribution
                # But scaling affects all entries in that term's support
                # Score: how much of the error can we fix without breaking other entries
                score = abs(u_val * v_val * w_val)
                if score > best_score:
                    best_score = score
                    best_term = t
            
            if best_term < 0 or best_score < 1e-15:
                # No good term found, try random perturbation
                scale = 0.01 / (1 + iteration * 0.1)
                U_try[:, 7:] += rng.normal(0, scale, (6, 9))
                V_try[:, 7:] += rng.normal(0, scale, (9, 9))
                W_try[:, 7:] += rng.normal(0, scale, (6, 9))
                continue
            
            # Adjust the best term to reduce the error at (a_max, b_max, c_max)
            t = best_term
            u_val = U_try[a_max, 7 + t]
            v_val = V_try[b_max, 7 + t]
            w_val = W_try[c_max, 7 + t]
            
            # Current contribution: u_val * v_val * w_val
            # Desired: u_val * v_val * w_val + err_val
            # Scale factor: 1 + err_val / (u_val * v_val * w_val)
            current = u_val * v_val * w_val
            if abs(current) < 1e-15:
                continue
            scale_factor = 1 + err_val / current
            
            # Apply scale to one of the three vectors (to minimize side effects)
            # Scale the W vector since it has the smallest support in correction terms
            # Actually, let's scale the entire term uniformly
            U_try[:, 7 + t] *= scale_factor ** (1/3)
            V_try[:, 7 + t] *= scale_factor ** (1/3)
            W_try[:, 7 + t] *= scale_factor ** (1/3)
            
            # Also add small random perturbations to escape local minima
            if iteration > 10:
                scale = 0.005 / (1 + (iteration - 10) * 0.05)
                U_try[:, 7:] += rng.normal(0, scale, (6, 9))
                V_try[:, 7:] += rng.normal(0, scale, (9, 9))
                W_try[:, 7:] += rng.normal(0, scale, (6, 9))
        
        if success:
            # Verify
            T_final = _tensor_from_matrices(U_try, V_try, W_try)
            if np.allclose(T_final, T, atol=1e-9):
                return {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}
    
    return None

def _try_linear_dependency_search(deadline):
    """
    Try to find a linear dependency among the 17 terms that allows dropping one.
    
    A rank-1 term is a vector in R^(6*9*6) = R^324 (flattened).
    If the 17 vectors are linearly dependent, we can drop one and rescale the others.
    But we need the sum to remain T, so we need a specific type of dependency.
    
    Actually, we need: sum_{r != r0} alpha_r * term_r = term_r0 for some r0 and alphas.
    Then we can replace term_r0 with the sum and absorb it into the other terms.
    """
    n, m, p = 2, 3, 3
    U17, V17, W17 = _make_rank17()
    
    # Flatten each term into a vector
    terms_flat = []
    for r in range(17):
        term = np.einsum('a,b,c->abc', U17[:,r], V17[:,r], W17[:,r])
        terms_flat.append(term.flatten())
    
    terms_flat = np.array(terms_flat)  # shape (17, 324)
    
    # Check if any term is a linear combination of the others
    # For each r0, solve: terms_flat[r0] = sum_{r != r0} alpha_r * terms_flat[r]
    # This is a linear system in alpha
    
    for r0 in range(17):
        if time.time() > deadline - 20:
            break
        
        # Build the matrix of other terms
        other_indices = [r for r in range(17) if r != r0]
        A = terms_flat[other_indices].T  # shape (324, 16)
        b = terms_flat[r0]  # shape (324,)
        
        # Solve least squares
        try:
            alpha, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        # Check if the solution is good
        reconstruction = A @ alpha
        err = np.max(np.abs(reconstruction - b))
        
        if err < 1e-9:
            # We found a linear dependency!
            # Now we need to check if the alpha values are "nice" (integers or half-integers)
            # For now, just use them
            # New terms: for r != r0, new_term_r = term_r + alpha_{r-r0} * term_r0
            # But this doesn't reduce rank; we need to absorb term_r0 into the others
            
            # Actually, if term_r0 = sum_{r != r0} alpha_r * term_r,
            # then T = sum_{r=0}^{16} term_r = sum_{r != r0} term_r + term_r0
            #     = sum_{r != r0} term_r + sum_{r != r0} alpha_r * term_r
            #     = sum_{r != r0} (1 + alpha_r) * term_r
            # So we can just scale the other terms!
            
            U_new = np.zeros((6, 16))
            V_new = np.zeros((9, 16))
            W_new = np.zeros((6, 16))
            
            for new_r, old_r in enumerate(other_indices):
                factor = 1 + alpha[new_r]
                U_new[:, new_r] = U17[:, old_r] * factor
                V_new[:, new_r] = V17[:, old_r] * factor
                W_new[:, new_r] = W17[:, old_r] * factor
            
            # Verify
            T_check = _tensor_from_matrices(U_new, V_new, W_new)
            T_true = _build_tensor(n, m, p)
            if np.allclose(T_check, T_true, atol=1e-9):
                return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    # First try linear dependency search (fast)
    result_ld = _try_linear_dependency_search(deadline)
    if result_ld is not None:
        return result_ld
    
    # Try structured rank-16 search
    result_struct = _try_structured_rank16(deadline)
    if result_struct is not None:
        return result_struct
    
    # Fall back to rank-17 solution
    U17, V17, W17 = _make_rank17()
    
    # Verify
    T = _build_tensor(n, m, p)
    if not _verify_decomposition(U17, V17, W17, T):
        rank_fb = n * m * p
        U_fb = np.zeros((n_a, rank_fb))
        V_fb = np.zeros((n_b, rank_fb))
        W_fb = np.zeros((n_c, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
# EVOLVE-BLOCK-END
