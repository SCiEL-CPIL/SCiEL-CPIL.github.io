"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
    
    Using GQ-style decomposition with merge search:
    - Split middle dim 4 = 2+2
    - For each 2x2 x 2x5 sub-problem, use Strassen for 2x2x2 and schoolbook for 2x2x1
    - Base rank: 2*(7+7+4) = 36
    - Search for mergeable term pairs to reduce rank
    """
    import time
    import numpy as np
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    rank_terms = []  # Each term is (u_dict, v_dict, w_dict)
    
    def add_strassen_2x2x2(a_offset, b_offset, c_offset, sign=1):
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset
        
        terms = [
            ({(i0, j0): 1, (i0+1, j0+1): 1}, 
             {(j0, k0): 1, (j0+1, k0+1): 1}, 
             {(i0, k0): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0): 1, (i0+1, j0+1): 1}, 
             {(j0, k0): 1}, 
             {(i0+1, k0): 1, (i0+1, k0+1): -1}),
            ({(i0, j0): 1}, 
             {(j0, k0+1): 1, (j0+1, k0+1): -1}, 
             {(i0, k0+1): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0+1): 1}, 
             {(j0+1, k0): 1, (j0, k0): -1}, 
             {(i0, k0): 1, (i0+1, k0): 1}),
            ({(i0, j0): 1, (i0, j0+1): 1}, 
             {(j0+1, k0+1): 1}, 
             {(i0, k0): -1, (i0, k0+1): 1}),
            ({(i0+1, j0): 1, (i0, j0): -1}, 
             {(j0, k0): 1, (j0, k0+1): 1}, 
             {(i0+1, k0+1): 1}),
            ({(i0, j0+1): 1, (i0+1, j0+1): -1}, 
             {(j0+1, k0): 1, (j0+1, k0+1): 1}, 
             {(i0, k0): 1}),
        ]
        
        for u, v, w in terms:
            u_scaled = {k: val * sign for k, val in u.items()}
            v_scaled = {k: val * sign for k, val in v.items()}
            w_scaled = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_scaled, v_scaled, w_scaled))
    
    def add_schoolbook_2x2x1(a_offset, b_offset, c_offset, sign=1):
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset
        
        terms = [
            ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
            ({(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1}),
            ({(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1}),
            ({(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1}),
        ]
        
        for u, v, w in terms:
            u_scaled = {k: val * sign for k, val in u.items()}
            v_scaled = {k: val * sign for k, val in v.items()}
            w_scaled = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_scaled, v_scaled, w_scaled))
    
    # Build the full decomposition
    add_strassen_2x2x2((0, 0), (0, 0), (0, 0))
    add_strassen_2x2x2((0, 0), (0, 2), (0, 2))
    add_schoolbook_2x2x1((0, 0), (0, 4), (0, 4))
    
    add_strassen_2x2x2((0, 2), (2, 0), (0, 0))
    add_strassen_2x2x2((0, 2), (2, 2), (0, 2))
    add_schoolbook_2x2x1((0, 2), (2, 4), (0, 4))
    
    base_rank = len(rank_terms)
    
    def terms_to_matrices(terms):
        """Convert list of (u_dict, v_dict, w_dict) to U, V, W matrices."""
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j, col] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k, col] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i, col] = coeff
        return U, V, W
    
    def verify_decomposition(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        # Check: sum_r U[:,r] * V[:,r]^T * W[:,r]^T should match the multiplication tensor
        # The tensor T[i,j,k] = delta_{i,j,k} for matrix mult is:
        # T[i,j,k] = 1 if the (i,k) entry of A*B involves A[i,j]*B[j,k]
        # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        # where i in [0,n), j in [0,m), k in [0,p)
        
        # Build the tensor from U, V, W
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Use einsum for efficiency
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # We need: for each (i,j,k), sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        rank = U.shape[1]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # T[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
        # This is a tensor contraction
        # Use einsum: 'ijr,jkr,ki r->ijk'
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        # Expected: T[i,j,k] = 1 for all valid i,j,k
        expected = np.ones((n, m, p))
        
        return np.allclose(T, expected, atol=1e-10)
    
    def try_merge(terms):
        """
        Try to find pairs of terms that can be merged (their sum is rank-1).
        Two rank-1 terms (u1,v1,w1) and (u2,v2,w2) can be merged if
        there exist scalars a,b such that a*u1 + b*u2, a*v1 + b*v2, a*w1 + b*w2
        are all rank-1 tensors (i.e., outer products).
        
        For our sparse terms, we check if the combined U, V, W vectors are
        all rank-1 (can be written as outer products).
        """
        if time.time() > deadline - 5:
            return None
        
        rank = len(terms)
        merged = []
        used = [False] * rank
        merges_found = 0
        
        for i in range(rank):
            if used[i]:
                continue
            if time.time() > deadline - 5:
                break
            for j in range(i+1, rank):
                if used[j]:
                    continue
                
                u1, v1, w1 = terms[i]
                u2, v2, w2 = terms[j]
                
                # Try to find scalars a, b such that:
                # a*u1 + b*u2 is rank-1 (as a vector in R^(n*m), it's always "rank-1" in the sense of being a vector)
                # Wait, U, V, W are vectors, not matrices. So any linear combination is still a vector.
                # The issue is different: we need the combined term to still contribute correctly.
                
                # Actually, two rank-1 terms can be merged if there exist scalars a, b, c (not all zero)
                # such that:
                # a*(u1⊗v1⊗w1) + b*(u2⊗v2⊗w2) = c*(u3⊗v3⊗w3) for some rank-1 tensor u3⊗v3⊗w3
                
                # This is equivalent to: the tensor a*u1⊗v1⊗w1 + b*u2⊗v2⊗w2 has rank 1.
                
                # For sparse vectors, let's check if the supports allow this.
                # If u1 and u2 have disjoint supports, and v1 and v2 have disjoint supports,
                # and w1 and w2 have disjoint supports, then the sum cannot be rank-1 unless one term is zero.
                
                # Let's check support overlap
                u1_keys = set(u1.keys())
                u2_keys = set(u2.keys())
                v1_keys = set(v1.keys())
                v2_keys = set(v2.keys())
                w1_keys = set(w1.keys())
                w2_keys = set(w2.keys())
                
                # If any pair has disjoint supports, merging is impossible (for non-trivial a,b)
                if not (u1_keys & u2_keys) or not (v1_keys & v2_keys) or not (w1_keys & w2_keys):
                    continue
                
                # Build the combined tensor and check if it's rank-1
                # Create sparse vectors
                all_u_keys = u1_keys | u2_keys
                all_v_keys = v1_keys | v2_keys
                all_w_keys = w1_keys | w2_keys
                
                # We need to check if there exist a, b such that the tensor is rank-1.
                # The tensor T = a*u1⊗v1⊗w1 + b*u2⊗v2⊗w2
                # For this to be rank-1, we need T = c*u3⊗v3⊗w3 for some u3, v3, w3.
                
                # This is a hard problem in general. Let's try a specific approach:
                # Assume the merged term has the form where we can factor out common structure.
                
                # Try: set a=1, solve for b and check if rank-1.
                # Build the tensor for a=1, b=b_val
                # T[i,j,k] = u1[i,j]*v1[i,j]*w1[i,j] + b_val*u2[i,j]*v2[i,j]*w2[i,j]
                
                # This is getting complex. Let's try a simpler heuristic:
                # Check if u1 is proportional to u2, v1 to v2, w1 to w2
                # If so, they're already the same direction and can be merged trivially.
                
                # Check proportionality
                def is_proportional(d1, d2):
                    if not d1 or not d2:
                        return False
                    keys1 = set(d1.keys())
                    keys2 = set(d2.keys())
                    if keys1 != keys2:
                        return False
                    # Check if all ratios are the same
                    ratios = []
                    for key in keys1:
                        if d2[key] == 0:
                            if d1[key] != 0:
                                return False
                            continue
                        ratio = d1[key] / d2[key]
                        if not ratios:
                            ratios.append(ratio)
                        elif abs(ratio - ratios[0]) > 1e-10:
                            return False
                    return True
                
                if is_proportional(u1, u2) and is_proportional(v1, v2) and is_proportional(w1, w2):
                    # They're in the same direction, merge them
                    merged_u = {}
                    for key in all_u_keys:
                        val = u1.get(key, 0) + u2.get(key, 0)
                        if val != 0:
                            merged_u[key] = val
                    merged_v = {}
                    for key in all_v_keys:
                        val = v1.get(key, 0) + v2.get(key, 0)
                        if val != 0:
                            merged_v[key] = val
                    merged_w = {}
                    for key in all_w_keys:
                        val = w1.get(key, 0) + w2.get(key, 0)
                        if val != 0:
                            merged_w[key] = val
                    
                    if merged_u and merged_v and merged_w:
                        merged.append((merged_u, merged_v, merged_w))
                        used[i] = True
                        used[j] = True
                        merges_found += 1
                        break  # Restart the outer loop since we modified the list
                    else:
                        used[i] = True
                        used[j] = True
                        merges_found += 1
                        break
                else:
                    continue
            
            # If we found a merge, restart
            if merges_found > 0:
                # Rebuild the terms list
                new_terms = []
                for idx in range(len(terms)):
                    if not used[idx]:
                        new_terms.append(terms[idx])
                new_terms.extend(merged)
                return new_terms
        
        return None
    
    # Try to merge terms
    current_terms = rank_terms[:]
    for _ in range(10):  # Try up to 10 merge iterations
        if time.time() > deadline - 10:
            break
        new_terms = try_merge(current_terms)
        if new_terms is None:
            break
        current_terms = new_terms
    
    final_terms = current_terms
    final_rank = len(final_terms)
    
    # Verify the decomposition
    U, V, W = terms_to_matrices(final_terms)
    if not verify_decomposition(U, V, W):
        # Fallback to original if verification fails
        U, V, W = terms_to_matrices(rank_terms)
    
    # Convert numpy arrays to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
