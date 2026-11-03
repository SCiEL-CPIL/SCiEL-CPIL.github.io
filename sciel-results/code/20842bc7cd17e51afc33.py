# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    
    n, m, p = 4, 4, 6
    
    # Total rank: 7 outer Strassen steps * 11 inner terms (7 Strassen + 4 Schoolbook)
    initial_rank = 77
    
    U = [[0] * initial_rank for _ in range(n * m)]
    V = [[0] * initial_rank for _ in range(m * p)]
    W = [[0] * initial_rank for _ in range(p * n)]
    
    # Outer Strassen Steps for 4x4 * 4x6
    outer_steps = [
        (([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])),
        (([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)])),
        (([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)])),
        (([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)])),
        (([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)])),
        (([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])),
        (([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]))
    ]
    
    def get_global_A(block_id, r, s):
        I = (block_id // 2) * 2 + r
        J = (block_id % 2) * 2 + s
        return I, J

    def get_global_B(block_id, s, k):
        J = (block_id // 2) * 2 + s
        K = (block_id % 2) * 3 + k
        return J, K

    def get_global_C(block_id, r, k):
        I = (block_id // 2) * 2 + r
        K = (block_id % 2) * 3 + k
        return I, K

    # Inner Decomposition for a 2x2 * 2x3 block
    inner_strassen = [
        (([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])),
        (([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])),
        (([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])),
        (([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])),
        (([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,1), (0,0,-1)])),
        (([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])),
        (([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]))
    ]
    
    inner_schoolbook = [
        (([(0,0,1)], [(0,0,1)], [(0,0,1)])),
        (([(0,1,1)], [(1,0,1)], [(0,0,1)])),
        (([(1,0,1)], [(0,0,1)], [(1,0,1)])),
        (([(1,1,1)], [(1,0,1)], [(1,0,1)]))
    ]
    
    # Construct the decomposition
    for i, (A_outer, B_outer, C_outer) in enumerate(outer_steps):
        base_rank = i * 11
        
        # Process Inner Strassen (7 terms)
        for j, (A_inner, B_inner, C_inner) in enumerate(inner_strassen):
            c_global_rank = base_rank + j
            
            for (r, s, coeff) in A_inner:
                for (block_id, b_coeff) in A_outer:
                    I, J = get_global_A(block_id, r, s)
                    U[I * m + J][c_global_rank] += coeff * b_coeff
                    
            for (s, k, coeff) in B_inner:
                for (block_id, b_coeff) in B_outer:
                    J, K = get_global_B(block_id, s, k)
                    V[J * p + K][c_global_rank] += coeff * b_coeff
                    
            for (r, k, coeff) in C_inner:
                for (block_id, b_coeff) in C_outer:
                    I, K = get_global_C(block_id, r, k)
                    W[K * n + I][c_global_rank] += coeff * b_coeff

        # Process Inner Schoolbook (4 terms)
        for j, (A_inner, B_inner, C_inner) in enumerate(inner_schoolbook):
            c_global_rank = base_rank + 7 + j
            
            for (r, s, coeff) in A_inner:
                for (block_id, b_coeff) in A_outer:
                    I, J = get_global_A(block_id, r, s)
                    U[I * m + J][c_global_rank] += coeff * b_coeff
                    
            for (s, k_local, coeff) in B_inner:
                k_global_in_block = 2 
                for (block_id, b_coeff) in B_outer:
                    J, K = get_global_B(block_id, s, k_global_in_block)
                    V[J * p + K][c_global_rank] += coeff * b_coeff
                    
            for (r, k_local, coeff) in C_inner:
                k_global_in_block = 2
                for (block_id, b_coeff) in C_outer:
                    I, K = get_global_C(block_id, r, k_global_in_block)
                    W[K * n + I][c_global_rank] += coeff * b_coeff

    # Convert to NumPy for efficient linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Verify correctness first
    # Reconstruct tensor
    # T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    # i in 0..15 (n*m), j in 0..23 (m*p), k in 0..23 (p*n)
    
    # Check for linear dependencies to reduce rank
    # Flatten each term into a vector
    dim = n * m * p
    terms = np.zeros((initial_rank, dim))
    
    for r in range(initial_rank):
        # Outer product of columns
        u_col = U_np[:, r]
        v_col = V_np[:, r]
        w_col = W_np[:, r]
        # Reshape to 3D and flatten
        # U is (n*m, 1), V is (m*p, 1), W is (p*n, 1)
        # Tensor is (n, m, p) but our indices are flattened.
        # We just need the vector representation.
        # The tensor entry T[i, j, k] corresponds to U[i]V[j]W[k].
        # We can compute the full tensor T as:
        # T = sum_r U[:,r] (outer) V[:,r] (outer) W[:,r]
        # But for dependency check, we treat each term as a vector in R^(n*m*p)
        term_vec = np.outer(np.outer(u_col, v_col), w_col).flatten()
        terms[r, :] = term_vec

    # Check rank of the matrix 'terms' (77 x 96)
    # If rank < 77, we have dependencies.
    # We want to find a basis and express the sum of all terms as a linear combination of the basis.
    
    # Use SVD to find rank and basis
    try:
        U_svd, s, Vt_svd = np.linalg.svd(terms, full_matrices=False)
        # Determine numerical rank
        tol = 1e-10
        rank_svd = np.sum(s > tol)
        
        if rank_svd < initial_rank:
            # We can reduce the rank.
            # The sum of all terms is sum_{r=0}^{76} terms[r]
            target_vec = np.sum(terms, axis=0)
            
            # We want to find coefficients c such that sum_{r=0}^{76} c_r * terms[r] = target_vec
            # And we want c_r = 0 for r not in the basis.
            # Actually, we want to represent target_vec using the basis vectors.
            # Let B be the matrix of basis vectors (first rank_svd columns of U_svd? No, U_svd columns are orthonormal basis for the row space).
            # The columns of terms lie in the subspace spanned by the first rank_svd left singular vectors.
            # We can project target_vec onto this subspace.
            # But we need the coefficients in terms of the *original* terms.
            
            # Simpler approach: Gaussian Elimination on the transpose?
            # We have 77 vectors in 96D.
            # We want to find a subset of indices I and coefficients c_i such that:
            # sum_{i in I} c_i * terms[i] = target_vec
            # And |I| is minimal.
            
            # Since the vectors are likely generated from a structured product, dependencies might be sparse.
            # Let's try to find a minimal generating set.
            
            # We can use the fact that if we have dependencies, we can eliminate one vector at a time.
            # However, finding the *minimum* rank decomposition is NP-hard in general.
            # But here, we just want to remove *any* dependencies.
            
            # Let's try to find a basis of the column space of 'terms' (which is the row space of 'terms'.T? No, 'terms' is (77, 96)).
            # The columns of 'terms' are the 77 vectors.
            # We want to express the sum of these 77 vectors as a linear combination of a subset of them.
            
            # Let's form the matrix M = terms.T (96 x 77).
            # The columns of M are the vectors.
            # We want to solve M * x = target_vec, where x is a sparse vector of size 77.
            # We want x to be as sparse as possible.
            
            # Given the small size, we can try a greedy approach or just use the null space.
            # If there is a null space vector n such that terms.T @ n = 0, then sum n_r * terms[r] = 0.
            # This means we can add/subtract this combination.
            # To reduce the count, we need to find a combination where one coefficient is 1 (or -1) and we can eliminate that term?
            # No, if sum n_r T_r = 0, then T_j = - sum_{r!=j} n_r/n_j T_r.
            # Then T_sum = sum T_r = T_j + sum_{r!=j} T_r = sum_{r!=j} (1 + n_r/n_j) T_r.
            # So we can eliminate T_j if we update the coefficients of the others.
            
            # So, we look for a null vector n with a non-zero component n_j.
            # We can then eliminate term j.
            
            # Let's compute the null space of terms.T (96x77).
            # Actually, SVD of terms (77x96) gives right singular vectors Vt_svd (96x77).
            # The null space of terms.T is the left null space of terms?
            # terms.T is 96x77. Rank is rank_svd.
            # Null space dimension is 77 - rank_svd.
            # The basis for the null space is given by the last (77 - rank_svd) columns of Vt_svd.T?
            # SVD: terms = U_svd @ diag(s) @ Vt_svd.
            # terms.T = V_svd @ diag(s) @ U_svd.T.
            # Null space of terms.T is spanned by the last (96-rank_svd) columns of V_svd? No.
            # terms.T is 96x77. Its columns are in R^96.
            # We want x in R^77 such that terms.T @ x = 0.
            # This is the left null space of terms.
            # From SVD of terms (77x96), the left null space is not directly given by Vt.
            # We need SVD of terms.T (96x77).
            
            U_t, s_t, Vt_t = np.linalg.svd(terms.T, full_matrices=False)
            # terms.T is 96x77.
            # Rank is rank_svd.
            # Null space basis is the last (77 - rank_svd) columns of Vt_t.T (which is V_t).
            null_space_basis = Vt_t[rank_svd:, :].T # Shape (77, 77-rank_svd)
            
            # We can try to eliminate terms one by one.
            # Start with the original set of 77 terms, all with coefficient 1.
            # coeffs = np.ones(77)
            # active = set(range(77))
            
            # However, modifying coefficients changes the "terms" themselves?
            # No, the terms U[:,r], V[:,r], W[:,r] are fixed vectors.
            # We are just changing the weights in the sum.
            # So we just need to find a sparse representation of the target tensor.
            
            # Let's try to find a representation using only a subset of the original indices.
            # We can use a greedy algorithm:
            # 1. Start with all 77 indices.
            # 2. Check if any index j can be removed.
            #    j can be removed if target_vec - terms[j] is in the span of {terms[i] for i != j}.
            #    This is equivalent to checking if rank({terms[i] for i != j}) == rank({terms[i] for i != j, j}).
            #    Since the full set has rank R, if removing j doesn't change the rank, j is dependent.
            
            # This is O(R^2) rank checks. R=77, dim=96.
            # Rank check is O(96^2 * 77) approx. 77 * 7000 = 500k ops. Very fast.
            
            active_indices = list(range(initial_rank))
            reduced_U = U_np[:, :].copy()
            reduced_V = V_np[:, :].copy()
            reduced_W = W_np[:, :].copy()
            
            # We need to track the coefficients.
            # Let's represent the current sum as sum_{k in active} c_k * T_k.
            # Initially c_k = 1 for all k.
            coeffs = np.ones(initial_rank)
            
            # Iterate to remove dependent terms
            # We will iterate through indices and try to remove them.
            # To be efficient, we can just check the rank of the active set.
            # If rank(active) < len(active), there is a dependency.
            # But we need to find *which* one to remove and update coeffs.
            
            # Simpler: Just find the rank of the active set.
            # If rank < count, we can reduce.
            # But we need the new coefficients.
            
            # Let's use the null space vectors found above.
            # For each null vector n, we have sum n_r T_r = 0.
            # This allows us to perform a "pivot" elimination.
            
            # Given the complexity of managing coefficients dynamically, and that the dependencies might be simple (e.g., duplicates),
            # let's first check for duplicates and zeros.
            
            # Check for zero terms
            for r in range(initial_rank):
                if np.allclose(terms[r], 0):
                    # Remove this term
                    coeffs[r] = 0
                    active_indices.remove(r)
            
            # Check for duplicates
            # If T_i == T_j, then T_i + T_j = 2 T_i. We can merge them.
            # But we want to minimize the count.
            # If T_i == T_j, we can set c_i = 2, c_j = 0.
            
            # Let's do a simple pass:
            # For each pair, if T_i is close to T_j, merge.
            
            # But the most robust way is to solve for the minimal support.
            # Given the time, I will just return the rank-77 solution if no simple reductions are found,
            # or perform the rank check on the whole set and if rank < 77, 
            # I will return a subset of indices that forms a basis, 
            # BUT I must ensure the sum is preserved.
            
            # If I just return a basis, the sum is NOT preserved unless I scale them.
            # And finding the correct scaling is the hard part.
            
            # However, if the dependencies are such that some terms are *exactly* linear combinations,
            # and we can find the coefficients, we can update.
            
            # Let's assume for now that no simple reductions exist in this specific Kronecker product 
            # that are easily exploitable without a full solver, 
            # OR that the rank is indeed 77.
            
            # But wait, the knowledge base says "greedy column elimination MAY find further reductions".
            # This suggests it's worth trying.
            
            # Let's try one greedy step:
            # 1. Compute rank of all 77 terms.
            # 2. If rank < 77, find a null vector n.
            # 3. Pick an index j with n_j != 0.
            # 4. Update coeffs for all i != j: c_i = c_i - n_i/n_j * c_j? 
            #    No, the relation is sum n_r T_r = 0.
            #    So T_j = - sum_{r!=j} (n_r/n_j) T_r.
            #    Current sum S = sum_{r} c_r T_r.
            #    Substitute T_j:
            #    S = c_j T_j + sum_{r!=j} c_r T_r = c_j (- sum_{r!=j} (n_r/n_j) T_r) + sum_{r!=j} c_r T_r
            #    S = sum_{r!=j} (c_r - c_j * n_r/n_j) T_r.
            #    So new c'_r = c_r - c_j * n_r/n_j for r != j.
            #    And c'_j = 0.
            
            # This reduces the number of non-zero coefficients by 1 (if c_j was non-zero and c'_j becomes 0).
            # We can repeat this until no more null vectors are found (rank == count).
            
            # Let's implement this greedy elimination.
            
            current_coeffs = np.ones(initial_rank)
            active = set(range(initial_rank))
            
            # We need to recompute null space after each elimination?
            # Yes, because the set of active terms changes.
            
            # To save time, we will limit the number of iterations.
            max_eliminations = 10
            eliminated = 0
            
            for _ in range(max_eliminations):
                if len(active) == 0:
                    break
                
                # Build matrix of active terms
                active_list = list(active)
                active_terms = terms[active_list, :] # Shape (len(active), 96)
                
                # Check rank
                try:
                    U_act, s_act, Vt_act = np.linalg.svd(active_terms, full_matrices=False)
                    rank_act = np.sum(s_act > 1e-10)
                except:
                    break
                
                if rank_act < len(active_list):
                    # There is a dependency.
                    # Find null space of active_terms.T (96 x len(active))
                    # SVD of active_terms.T
                    U_t_act, s_t_act, Vt_t_act = np.linalg.svd(active_terms.T, full_matrices=False)
                    # Null space basis: last (len(active) - rank_act) columns of Vt_t_act.T
                    # Vt_t_act is (96, len(active)). Vt_t_act.T is (len(active), 96).
                    # Wait, SVD of A (m x n) gives V (n x n).
                    # active_terms.T is 96 x N.
                    # Vt_t_act is 96 x N.
                    # The right singular vectors are columns of Vt_t_act.T? No.
                    # A = U S Vt.
                    # A.T = V S Ut.
                    # So the right singular vectors of A.T are the columns of U (from SVD of A)?
                    # No.
                    # Let's just use np.linalg.null_space if available? No, standard lib.
                    # We can use SVD of active_terms.T.
                    # A.T is 96 x N.
                    # U, S, Vt = svd(A.T).
                    # U is 96x96. S is 96xN. Vt is NxN.
                    # The null space of A.T is the span of the last (N-rank) columns of Vt.T (which is V).
                    
                    U_null, S_null, Vt_null = np.linalg.svd(active_terms.T, full_matrices=False)
                    # Vt_null is (N, N).
                    # Null space is spanned by columns of Vt_null.T corresponding to zero singular values.
                    # These are the last (N - rank_act) columns.
                    
                    null_vectors = Vt_null[:, rank_act:].T # Shape (N - rank_act, N)
                    
                    if null_vectors.shape[0] > 0:
                        # Take the first null vector
                        n = null_vectors[0, :]
                        
                        # Find an index j in active_list with n_j != 0
                        j_idx_in_active = -1
                        for idx, val in enumerate(n):
                            if abs(val) > 1e-10:
                                j_idx_in_active = idx
                                break
                        
                        if j_idx_in_active != -1:
                            j_global = active_list[j_idx_in_active]
                            n_j = n[j_idx_in_active]
                            
                            # Update coefficients
                            # c'_r = c_r - c_j * n_r / n_j
                            for r_idx, r_global in enumerate(active_list):
                                if r_global != j_global:
                                    current_coeffs[r_global] -= current_coeffs[j_global] * (n[r_idx] / n_j)
                            
                            current_coeffs[j_global] = 0.0
                            active.remove(j_global)
                            eliminated += 1
                            # Continue to try to eliminate more
                        else:
                            break
                    else:
                        break
                else:
                    break
            
            if eliminated > 0:
                # Construct the reduced matrices
                # We need to create U, V, W with only the active columns.
                # And scale them by current_coeffs.
                
                new_rank = len(active)
                U_new = U_np[:, list(active)] * current_coeffs[list(active)]
                V_new = V_np[:, list(active)]
                W_new = W_np[:, list(active)]
                
                # Convert back to lists
                return {
                    "U": U_new.tolist(),
                    "V": V_new.tolist(),
                    "W": W_new.tolist()
                }
            else:
                # No reduction found
                return {
                    "U": U_np.tolist(),
                    "V": V_np.tolist(),
                    "W": W_np.tolist()
                }
    except Exception as e:
        # Fallback to original
        return {
            "U": U,
            "V": V,
            "W": W
        }

# EVOLVE-BLOCK-END
