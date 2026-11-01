import time
import numpy as np

DEADLINE = 115.0

def construct():
    start_time = time.time()
    n, m, p = 2, 5, 6
    total_entries = n * m * p  # 60
    
    # Build a large pool of candidate rank-1 tensors
    # A rank-1 tensor is (u_vec, v_vec, w_vec) where:
    #   u_vec has length n*m, v_vec has length m*p, w_vec has length p*n
    # and the contribution to the (i,j,k) entry is u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    candidates = []  # list of (u_vec, v_vec, w_vec, contrib_vec)
    
    def add_candidate(u_sparse, v_sparse, w_sparse):
        """Add a candidate defined by sparse dicts."""
        u_vec = np.zeros(n * m)
        v_vec = np.zeros(m * p)
        w_vec = np.zeros(p * n)
        
        for (i, j), val in u_sparse.items():
            u_vec[i * m + j] = val
        for (j, k), val in v_sparse.items():
            v_vec[j * p + k] = val
        for (k, i), val in w_sparse.items():
            w_vec[k * n + i] = val
        
        # Compute contribution vector
        contrib = np.zeros(total_entries)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    contrib[i * m * p + j * p + k] = u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
        
        candidates.append((u_vec, v_vec, w_vec, contrib))
    
    # Type 1: Standard unit tensors
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_candidate({(i,j): 1}, {(j,k): 1}, {(k,i): 1})
    
    # Type 2: Strassen 2x2x2 blocks
    def strassen_terms(io, jo, ko):
        terms = []
        terms.append(({(io,jo): 1, (io+1,jo+1): 1}, 
                      {(jo,ko): 1, (jo+1,ko+1): 1}, 
                      {(ko,io): 1, (ko+1,io+1): 1}))
        terms.append(({(io+1,jo): 1, (io+1,jo+1): 1}, 
                      {(jo,ko): 1}, 
                      {(ko,io+1): 1, (ko+1,io+1): -1}))
        terms.append(({(io,jo): 1}, 
                      {(jo,ko+1): 1, (jo+1,ko+1): -1}, 
                      {(ko+1,io): 1, (ko+1,io+1): 1}))
        terms.append(({(io+1,jo+1): 1}, 
                      {(jo+1,ko): 1, (jo,ko): -1}, 
                      {(ko,io): 1, (ko,io+1): 1}))
        terms.append(({(io,jo): 1, (io,jo+1): 1}, 
                      {(jo+1,ko+1): 1}, 
                      {(ko,io): -1, (ko+1,io): 1}))
        terms.append(({(io+1,jo): 1, (io,jo): -1}, 
                      {(jo,ko): 1, (jo,ko+1): 1}, 
                      {(ko+1,io+1): 1}))
        terms.append(({(io,jo+1): 1, (io+1,jo+1): -1}, 
                      {(jo+1,ko): 1, (jo+1,ko+1): 1}, 
                      {(ko,io): 1}))
        return terms
    
    for io in range(n - 1):
        for jo in range(m - 1):
            for ko in range(p - 1):
                for u_c, v_c, w_c in strassen_terms(io, jo, ko):
                    add_candidate(u_c, v_c, w_c)
    
    # Type 3: Rank-11 bilinear algorithm for 2x3x3 sub-blocks
    # We can apply a rank-11 algorithm to 2x3x3 blocks
    # For 2x3x3: rows i=io,io+1; cols j=jo..jo+2; k=ko..ko+2
    # The rank of the 2x3x3 multiplication tensor is 11 (not 18)
    # This is a known result from algebraic complexity theory
    
    def rank11_2x3x3_terms(io, jo, ko):
        """
        Return 11 rank-1 terms that decompose the 2x3x3 matrix multiplication tensor.
        This uses a construction based on the fact that the rank of M_{2,3,3} is 11.
        """
        terms = []
        # i indices: io, io+1
        # j indices: jo, jo+1, jo+2
        # k indices: ko, ko+1, ko+2
        
        # Construction: Use the rank-7 for 2x2x2 on the first two j's,
        # plus additional terms for the third j and interactions.
        # Actually, let me use a known construction.
        
        # A simpler approach: Use rank 7 for 2x2x2 on (i, j=jo..jo+1, k=ko..ko+2)
        # But that doesn't directly work since Strassen is for 2x2x2.
        
        # Let me use a different strategy:
        # The 2x3x3 tensor can be seen as 2x2x3 + 2x1x3
        # 2x2x3 has rank 11? No, let me think differently.
        
        # Actually, let me just use the known fact that we can get rank 11
        # by combining Strassen-like terms with the extra column.
        
        # For now, let me generate candidates that involve all three j's:
        
        # Term 1: (A00+A11)(B00+B11+B22)
        u1 = {(io,jo): 1, (io+1,jo+2): 1}
        v1 = {(jo,ko): 1, (jo+1,ko+1): 1, (jo+2,ko+2): 1}
        w1 = {(ko,io): 1, (ko+1,io+1): 1, (ko+2,io+1): 1}
        terms.append((u1, v1, w1))
        
        # Term 2: (A00+A11)(B00-B11-B22)
        u2 = {(io,jo): 1, (io+1,jo+2): 1}
        v2 = {(jo,ko): 1, (jo+1,ko+1): -1, (jo+2,ko+2): -1}
        w2 = {(ko,io): 1, (ko+1,io+1): 1, (ko+2,io+1): 1}
        terms.append((u2, v2, w2))
        
        # Term 3: (A01+A12)(B00+B11)
        u3 = {(io,jo+1): 1, (io+1,jo+2): 1}
        v3 = {(jo,ko): 1, (jo+1,ko+1): 1}
        w3 = {(ko,io): 1, (ko+1,io+1): 1}
        terms.append((u3, v3, w3))
        
        # Term 4: (A01-A12)(B00-B11)
        u4 = {(io,jo+1): 1, (io+1,jo+2): -1}
        v4 = {(jo,ko): 1, (jo+1,ko+1): -1}
        w4 = {(ko,io): 1, (ko+1,io+1): 1}
        terms.append((u4, v4, w4))
        
        # Term 5: A00*(B00+B11)
        u5 = {(io,jo): 1}
        v5 = {(jo,ko): 1, (jo+1,ko+1): 1}
        w5 = {(ko,io): 1, (ko+1,io): 1}
        terms.append((u5, v5, w5))
        
        # Term 6: A00*(B00-B11)
        u6 = {(io,jo): 1}
        v6 = {(jo,ko): 1, (jo+1,ko+1): -1}
        w6 = {(ko,io): 1, (ko+1,io): 1}
        terms.append((u6, v6, w6))
        
        # Term 7: A11*(B00+B11)
        u7 = {(io+1,jo+1): 1}
        v7 = {(jo,ko): 1, (jo+1,ko+1): 1}
        w7 = {(ko,io+1): 1, (ko+1,io+1): 1}
        terms.append((u7, v7, w7))
        
        # Term 8: A11*(B00-B11)
        u8 = {(io+1,jo+1): 1}
        v8 = {(jo,ko): 1, (jo+1,ko+1): -1}
        w8 = {(ko,io+1): 1, (ko+1,io+1): 1}
        terms.append((u8, v8, w8))
        
        # Term 9: A02*B22
        u9 = {(io,jo+2): 1}
        v9 = {(jo+2,ko+2): 1}
        w9 = {(ko+2,io): 1}
        terms.append((u9, v9, w9))
        
        # Term 10: A12*B22
        u10 = {(io+1,jo+2): 1}
        v10 = {(jo+2,ko+2): 1}
        w10 = {(ko+2,io+1): 1}
        terms.append((u10, v10, w10))
        
        # Term 11: Cross terms
        u11 = {(io,jo+1): 1, (io+1,jo): -1}
        v11 = {(jo,ko+2): 1, (jo+1,ko+1): -1}
        w11 = {(ko+2,io): 1, (ko+1,io+1): 1}
        terms.append((u11, v11, w11))
        
        return terms
    
    # Actually, the above construction is not verified to give rank 11.
    # Let me use a simpler approach: just generate more diverse candidates.
    
    # Type 3: Mixed patterns with 3 columns
    for io in range(n - 1):
        for jo in range(m - 2):
            for ko in range(p - 2):
                # Generate some structured candidates for 2x3x3
                # Simple: use standard terms for all 9 positions
                for j_off in range(3):
                    for k_off in range(3):
                        for i_off in range(2):
                            add_candidate(
                                {(io+i_off, jo+j_off): 1},
                                {(jo+j_off, ko+k_off): 1},
                                {(ko+k_off, io+i_off): 1}
                            )
    
    # Type 4: Additional Strassen-like terms with 3-wide blocks
    # These are heuristic candidates that might help
    
    # Now use greedy to find a good decomposition
    T_target = np.ones(total_entries)
    residual = T_target.copy()
    
    selected_indices = []
    selected_coeffs = []
    
    num_candidates = len(candidates)
    
    # Precompute all contribution vectors as a matrix
    contrib_matrix = np.zeros((num_candidates, total_entries))
    for idx in range(num_candidates):
        contrib_matrix[idx] = candidates[idx][3]
    
    # Precompute norms
    norms = np.sum(contrib_matrix ** 2, axis=1)
    
    # Precompute Gram matrix (candidate dot products)
    # This is num_candidates x num_candidates, might be large but let's try
    # For 2x5x6: candidates include 60 unit + ~42*7 strassen + more
    # Let's cap: if too many, skip full Gram
    if num_candidates <= 2000:
        Gram = contrib_matrix @ contrib_matrix.T
    else:
        Gram = None
    
    max_iter = 100
    for _ in range(max_iter):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        best_idx = -1
        best_reduction = 0
        best_coeff = 0
        
        if Gram is not None:
            # Fast path: precompute dot(residual, all_candidates) once
            dots = contrib_matrix @ residual
            valid = norms > 1e-12
            reductions = np.where(valid, dots**2 / np.where(norms>1e-12, norms, 1), 0)
            best_idx = int(np.argmax(reductions))
            best_reduction = reductions[best_idx]
            if best_reduction < 1e-9:
                break
            best_coeff = dots[best_idx] / norms[best_idx]
        else:
            for idx in range(num_candidates):
                if norms[idx] < 1e-12:
                    continue
                dot_rc = np.dot(residual, contrib_matrix[idx])
                coeff = dot_rc / norms[idx]
                reduction = (dot_rc ** 2) / norms[idx]
                
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_idx = idx
                    best_coeff = coeff
        
        if best_idx == -1 or best_reduction < 1e-9:
            break
        
        selected_indices.append(best_idx)
        selected_coeffs.append(best_coeff)
        residual -= best_coeff * contrib_matrix[best_idx]
        
        if np.linalg.norm(residual) < 1e-9:
            break
    
    # Rank reduction: find linear dependencies among selected terms
    def try_reduce_rank(sel_indices, sel_coeffs, U_arr, V_arr, W_arr):
        """Try to find linear dependencies among the rank-1 terms to reduce rank."""
        r = len(sel_indices)
        if r < 2:
            return None
        
        # Build the U, V, W matrices for these terms
        U_sel = U_arr.copy()
        V_sel = V_arr.copy()
        W_sel = W_arr.copy()
        
        # For each term r, we have vectors u_r, v_r, w_r
        # A linear dependency means: sum_r c_r * u_r ⊗ v_r ⊗ w_r = 0
        # This is equivalent to: the tensor T = sum c_r * outer(u_r, outer(v_r, w_r)) = 0
        # 
        # One approach: check if any term can be expressed as a combination of others
        # by solving a linear system in the "coefficient space"
        
        # Alternative: use the fact that if we have terms where the product
        # u_r[i]*v_r[j]*w_r[k] sums to zero for all (i,j,k), we can remove one.
        
        # Let's try: for each pair (r1, r2), check if term r2 is a scalar multiple
        # of term r1 in the tensor sense (i.e., all three vectors are proportional)
        # This is a simple check that might catch some redundancies.
        
        # More general: check if the set of tensors {u_r ⊗ v_r ⊗ w_r} is linearly dependent
        # by forming a matrix whose columns are the vectorized tensors and finding null space.
        
        # Vectorize each tensor
        n_entries = n * m * p
        tensor_matrix = np.zeros((n_entries, r))
        for col in range(r):
            idx = sel_indices[col]
            u_vec, v_vec, w_vec, _ = candidates[idx]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor_matrix[i*m*p + j*p + k, col] = u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
        
        # Find null space
        try:
            U_sv, s_vals, Vt = np.linalg.svd(tensor_matrix, full_matrices=False)
            # Null space vectors are rows of Vt corresponding to zero singular values
            tol = 1e-10 * (s_vals[0] if len(s_vals) > 0 else 1)
            null_mask = s_vals < tol
            if not np.any(null_mask):
                return None
            
            # Get null space basis
            null_space = Vt[null_mask].T  # shape (r, num_null)
            
            # Try to use each null space vector to eliminate one term
            best_reduced = None
            for ns_vec in null_space:
                # ns_vec has r components; find the component with largest magnitude
                # We can eliminate that term by expressing it as a combination of others
                max_idx = int(np.argmax(np.abs(ns_vec)))
                if abs(ns_vec[max_idx]) < 1e-10:
                    continue
                
                # New coefficients: for term t != max_idx, new_coeff[t] = old_coeff[t] - (ns_vec[t]/ns_vec[max_idx]) * old_coeff[max_idx]
                # Wait, that's not quite right. The null space vector means:
                # sum_t ns_vec[t] * tensor_t = 0
                # So tensor[max_idx] = -sum_{t != max_idx} (ns_vec[t]/ns_vec[max_idx]) * tensor_t
                # 
                # If we have coefficients c_t for each term, and we want to eliminate term max_idx:
                # The total tensor is sum_t c_t * tensor_t
                # = c_max * tensor_max + sum_{t != max} c_t * tensor_t
                # = c_max * (-sum_{t != max} (ns_vec[t]/ns_vec[max_idx]) * tensor_t) + sum_{t != max} c_t * tensor_t
                # = sum_{t != max} (c_t - c_max * ns_vec[t]/ns_vec[max_idx]) * tensor_t
                
                new_coeffs = list(sel_coeffs)
                c_max = new_coeffs[max_idx]
                for t in range(r):
                    if t != max_idx:
                        new_coeffs[t] = new_coeffs[t] - c_max * ns_vec[t] / ns_vec[max_idx]
                
                # Verify: the new set should give the same tensor
                # New tensor = sum_{t != max_idx} new_coeffs[t] * tensor_t
                # = sum_{t != max_idx} (c_t - c_max * ns_vec[t]/ns_vec[max_idx]) * tensor_t
                # = sum_{t != max_idx} c_t * tensor_t - c_max/ns_vec[max_idx] * sum_{t != max_idx} ns_vec[t] * tensor_t
                # = (sum_{t != max_idx} c_t * tensor_t + c_max * tensor_max) - c_max/ns_vec[max_idx] * (-ns_vec[max_idx] * tensor_max)
                # Wait, from null space: sum_t ns_vec[t] * tensor_t = 0
                # So sum_{t != max} ns_vec[t] * tensor_t = -ns_vec[max] * tensor_max
                # Thus: = sum_{t != max} c_t * tensor_t + c_max * tensor_max - c_max/ns_vec[max] * (-ns_vec[max] * tensor_max)
                # = sum_{t != max} c_t * tensor_t + c_max * tensor_max + c_max * tensor_max
                # That's wrong. Let me redo.
                
                # From null: sum_t ns[t] * T_t = 0, so ns[max]*T_max = -sum_{t!=max} ns[t]*T_t
                # T_max = -sum_{t!=max} (ns[t]/ns[max])*T_t
                # 
                # Original: sum_t c[t]*T_t = c[max]*T_max + sum_{t!=max} c[t]*T_t
                # = c[max]*(-sum_{t!=max} (ns[t]/ns[max])*T_t) + sum_{t!=max} c[t]*T_t
                # = sum_{t!=max} (c[t] - c[max]*ns[t]/ns[max])*T_t
                # 
                # So new_coeffs[t] = c[t] - c[max]*ns[t]/ns[max] for t != max
                # This is correct!
                
                # Build reduced matrices
                new_r = r - 1
                U_new = np.zeros((n * m, new_r))
                V_new = np.zeros((m * p, new_r))
                W_new = np.zeros((p * n, new_r))
                
                new_sel_indices = []
                col = 0
                for t in range(r):
                    if t == max_idx:
                        continue
                    new_sel_indices.append(sel_indices[t])
                    U_new[:, col] = U_sel[:, t]
                    V_new[:, col] = V_sel[:, t]
                    W_new[:, col] = W_sel[:, t]
                    col += 1
                
                # Verify
                A_test = np.random.randint(-10, 10, (n, m))
                B_test = np.random.randint(-10, 10, (m, p))
                C_test = A_test @ B_test
                
                C_calc = np.zeros((n, p))
                for i in range(n):
                    for k in range(p):
                        val = 0
                        for j in range(m):
                            for rr in range(new_r):
                                val += U_new[i*m+j, rr] * V_new[j*p+k, rr] * W_new[k*n+i, rr]
                        C_calc[i, k] = val
                
                if np.allclose(C_test, C_calc, atol=1e-5):
                    if best_reduced is None or new_r < best_reduced[0]:
                        best_reduced = (new_r, U_new, V_new, W_new)
            
            return best_reduced
        except np.linalg.LinAlgError:
            return None
    
    # Check if we got an exact solution
    if np.linalg.norm(residual) < 1e-9:
        rank_found = len(selected_indices)
        
        U = np.zeros((n * m, rank_found))
        V = np.zeros((m * p, rank_found))
        W = np.zeros((p * n, rank_found))
        
        for r, idx in enumerate(selected_indices):
            coeff = selected_coeffs[r]
            u_vec, v_vec, w_vec, _ = candidates[idx]
            U[:, r] = u_vec * coeff
            V[:, r] = v_vec
            W[:, r] = w_vec
        
        # Try rank reduction
        reduced = try_reduce_rank(selected_indices, selected_coeffs, U, V, W)
        if reduced is not None:
            new_rank, U, V, W = reduced
            rank_found = new_rank
        
        # Verify
        A_test = np.random.randint(-10, 10, (n, m))
        B_test = np.random.randint(-10, 10, (m, p))
        C_test = A_test @ B_test
        
        C_calc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0
                for j in range(m):
                    for r in range(rank_found):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_calc[i, k] = val
        
        if np.allclose(C_test, C_calc, atol=1e-5):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: return the known 54-rank solution
    n, m, p = 2, 5, 6
    rank_new = 54
    U_new = np.zeros((n * m, rank_new))
    V_new = np.zeros((m * p, rank_new))
    W_new = np.zeros((p * n, rank_new))
    
    current_rank = 0
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal current_rank
        # r=0: M1
        U_new[(i_off)*m + (j_off), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        # r=1: M2
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = -1
        current_rank += 1
        # r=2: M3
        U_new[(i_off)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = -1
        W_new[(k_off+1)*n + (i_off), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        # r=3: M4
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = -1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        # r=4: M5
        U_new[(i_off)*m + (j_off), current_rank] = 1
        U_new[(i_off)*m + (j_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = -1
        W_new[(k_off+1)*n + (i_off), current_rank] = 1
        current_rank += 1
        # r=5: M6
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        U_new[(i_off)*m + (j_off), current_rank] = -1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        V_new[(j_off)*p + (k_off+1), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        # r=6: M7
        U_new[(i_off)*m + (j_off+1), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = -1
        V_new[(j_off+1)*p + (k_off), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        current_rank += 1

    def add_standard_block(i_off, j_off, k_off):
        nonlocal current_rank
        U_new[(i_off)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        current_rank += 1
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        current_rank += 1

    # Part 1: A[:, 0:2] * B[0:2, :]
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # Part 2a: A[:, 2:3] * B[2:3, :]
    for k in range(6):
        add_standard_block(0, 2, k)
        
    # Part 2b: A[:, 3:5] * B[3:5, :]
    add_strassen_block(0, 3, 0)
    add_strassen_block(0, 3, 2)
    add_strassen_block(0, 3, 4)
    
    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
