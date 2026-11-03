import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140
    
    def peel_terms(U_vecs, V_vecs, W_vecs, n, m, p, deadline):
        """Iteratively peel off linearly dependent rank-1 terms."""
        rank = len(U_vecs)
        if rank <= 1:
            return U_vecs, V_vecs, W_vecs
        
        # Convert to numpy float arrays for linear algebra
        U_arr = np.array([[float(x) for x in row] for row in U_vecs]).T  # (n*m, rank)
        V_arr = np.array([[float(x) for x in row] for row in V_vecs]).T  # (m*p, rank)
        W_arr = np.array([[float(x) for x in row] for row in W_vecs]).T  # (p*n, rank)
        
        # Try to find linear dependencies and eliminate terms
        # Strategy: for each term, check if it can be expressed as a linear combination
        # of others. If so, remove it and adjust the others.
        
        # More practical: try pairwise combination
        # If term i and term j have proportional U vectors (or V or W), we can merge them.
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            rank = len(U_vecs)
            
            # Check for duplicate or proportional terms
            # Two terms can be merged if their U, V, W vectors are all proportional
            # Actually, a simpler approach: if U_i is a scalar multiple of U_j,
            # and V_i is a scalar multiple of V_j, and W_i is a scalar multiple of W_j,
            # then we can combine them.
            
            # Even simpler: if two terms are identical, remove one.
            # Or if U_i = c * U_j, we can try to eliminate term i.
            
            # Let's try: for each pair (i, j) where i > 0, check if U_i is proportional to U_j
            # If U_i = c * U_j, then we can write term i as c * (U_j, V_i/c, W_i/c)
            # This doesn't directly help unless V_i/c and W_i/c also relate to term j.
            
            # Better approach: Gaussian elimination on the U matrix
            # Find the rank of U matrix. If rank < number of terms, there are dependencies.
            
            U_mat = np.array([[float(x) for x in row] for row in U_vecs]).T  # (n*m, rank)
            V_mat = np.array([[float(x) for x in row] for row in V_vecs]).T  # (m*p, rank)
            W_mat = np.array([[float(x) for x in row] for row in W_vecs]).T  # (p*n, rank)
            
            # Find null space of U_mat
            try:
                U_rank = np.linalg.matrix_rank(U_mat)
                if U_rank < rank:
                    # There are linear dependencies in U
                    # Find null space
                    U_T = U_mat.T  # (rank, n*m)
                    # SVD to find null space
                    U_svd = np.linalg.svd(U_T, full_matrices=True)
                    U_Vt = U_svd[2]  # (rank, rank)
                    # Null space vectors are the last (rank - U_rank) rows of Vt
                    null_start = U_rank
                    null_vecs = U_Vt[null_start:]  # (rank - U_rank, rank)
                    
                    # For each null vector, we have a linear combination of terms that sums to zero in U
                    # This means: sum_t null_vec[t] * U_t = 0
                    # We can use this to eliminate one term.
                    
                    for nv in null_vecs:
                        if time.time() > deadline:
                            return U_vecs, V_vecs, W_vecs
                        
                        # Find the term with largest absolute coefficient to eliminate
                        max_idx = np.argmax(np.abs(nv))
                        if abs(nv[max_idx]) < 1e-10:
                            continue
                        
                        # We want to express term max_idx in terms of others
                        # nv[max_idx] * U[max_idx] = -sum_{t != max_idx} nv[t] * U[t]
                        # U[max_idx] = -sum_{t != max_idx} (nv[t] / nv[max_idx]) * U[t]
                        
                        # But we also need the V and W components to match
                        # This only works if the V and W components are also consistent
                        # Let's check: does the same linear combination work for V and W?
                        
                        # Check V: sum_t nv[t] * V[t] should be 0 (approximately)
                        V_check = np.sum(nv[:, None] * V_mat, axis=1)
                        W_check = np.sum(nv[:, None] * W_mat, axis=1)
                        
                        if np.max(np.abs(V_check)) > 1e-8 or np.max(np.abs(W_check)) > 1e-8:
                            continue
                        
                        # Great! We can eliminate term max_idx
                        # New terms: for t != max_idx, multiply term t by 1/nv[max_idx]... 
                        # Actually, the relation is:
                        # sum_t nv[t] * (U[t], V[t], W[t]) = (0, 0, 0)
                        # So: nv[max_idx] * (U[max_idx], V[max_idx], W[max_idx]) = 
                        #     -sum_{t != max_idx} nv[t] * (U[t], V[t], W[t])
                        # (U[max_idx], V[max_idx], W[max_idx]) = 
                        #     -sum_{t != max_idx} (nv[t] / nv[max_idx]) * (U[t], V[t], W[t])
                        
                        # We can remove term max_idx and adjust the others
                        # New coefficient for term t (t != max_idx): 1 - nv[t]/nv[max_idx] * 0... 
                        # Actually, we just remove term max_idx and scale the others
                        # Wait, that's not right. The tensor is:
                        # T = sum_t U[t] ⊗ V[t] ⊗ W[t]
                        # If sum_t nv[t] * U[t] ⊗ V[t] ⊗ W[t] = 0, then we can remove any term
                        # and adjust the others.
                        
                        # Remove term max_idx
                        new_U = [U_vecs[t] for t in range(rank) if t != max_idx]
                        new_V = [V_vecs[t] for t in range(rank) for _ in [0]]
                        new_V = [V_vecs[t] for t in range(rank) if t != max_idx]
                        new_W = [W_vecs[t] for t in range(rank) if t != max_idx]
                        
                        # Verify the new decomposition is still valid
                        # The removed term was: U[max_idx] ⊗ V[max_idx] ⊗ W[max_idx]
                        # And we know: U[max_idx] ⊗ V[max_idx] ⊗ W[max_idx] = 
                        #   -sum_{t != max_idx} (nv[t]/nv[max_idx]) * U[t] ⊗ V[t] ⊗ W[t]
                        # So the new sum is:
                        # sum_{t != max_idx} U[t] ⊗ V[t] ⊗ W[t] = 
                        #   T - U[max_idx] ⊗ V[max_idx] ⊗ W[max_idx] + U[max_idx] ⊗ V[max_idx] ⊗ W[max_idx] = T
                        # Wait, that's not right either. Let me think again.
                        
                        # Original: T = sum_{all t} U[t] ⊗ V[t] ⊗ W[t]
                        # We have: sum_{all t} nv[t] * U[t] ⊗ V[t] ⊗ W[t] = 0
                        # So: nv[max_idx] * U[max_idx] ⊗ V[max_idx] ⊗ W[max_idx] = 
                        #   -sum_{t != max_idx} nv[t] * U[t] ⊗ V[t] ⊗ W[t]
                        # U[max_idx] ⊗ V[max_idx] ⊗ W[max_idx] = 
                        #   -sum_{t != max_idx} (nv[t]/nv[max_idx]) * U[t] ⊗ V[t] ⊗ W[t]
                        
                        # New decomposition: T' = sum_{t != max_idx} U[t] ⊗ V[t] ⊗ W[t]
                        # T' = T - U[max_idx] ⊗ V[max_idx] ⊗ W[max_idx]
                        #    = T + sum_{t != max_idx} (nv[t]/nv[max_idx]) * U[t] ⊗ V[t] ⊗ W[t]
                        #    = sum_{t != max_idx} (1 + nv[t]/nv[max_idx]) * U[t] ⊗ V[t] ⊗ W[t]
                        
                        # So we need to scale each remaining term by (1 + nv[t]/nv[max_idx])
                        scale_factors = []
                        for t in range(rank):
                            if t == max_idx:
                                continue
                            scale = 1 + nv[t] / nv[max_idx]
                            if abs(scale) < 1e-10:
                                # This term can be removed entirely
                                continue
                            scale_factors.append((t, scale))
                        
                        if len(scale_factors) < rank - 1:
                            # We removed at least one more term
                            final_U = []
                            final_V = []
                            final_W = []
                            for t, scale in scale_factors:
                                final_U.append([x * scale for x in U_vecs[t]])
                                final_V.append([x * scale for x in V_vecs[t]])
                                final_W.append([x * scale for x in W_vecs[t]])
                            
                            # Verify
                            valid = True
                            for ci in range(n):
                                for cj in range(m):
                                    for ck in range(p):
                                        prod_sum = sum(
                                            final_U[t][ci*m + cj] * final_V[t][cj*p + ck] * final_W[t][ck*n + ci]
                                            for t in range(len(final_U))
                                        )
                                        if abs(prod_sum - 1) > 1e-6:
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                if not valid:
                                    break
                            
                            if valid:
                                U_vecs = final_U
                                V_vecs = final_V
                                W_vecs = final_W
                                changed = True
                                break
                            else:
                                # Revert
                                continue
                        else:
                            continue
            except:
                pass
        
        return U_vecs, V_vecs, W_vecs
    n, m, p = 3, 4, 7

    # Strassen 2x2x2 decomposition (7 terms)
    # Each term: (u_mask, v_mask, w_mask) where masks are 2-element tuples
    # u_mask[i] = coefficient for U[i*m+j] terms
    # We represent each rank-1 term as (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    # for the 2x2x2 block at offsets (i0, j0, k0)
    
    strassen_terms = [
        # Term 0: (a+c)(e+g)
        ([0, 3], [1, 1], [0, 5], [1, 1], [0, 3], [1, 1]),
        # Term 1: (b+d)(e)
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),
        # Term 2: (a)(f-g)
        ([0], [1], [1, 5], [1, -1], [1, 3], [1, 1]),
        # Term 3: (d)(g-f)
        ([3], [1], [5, 1], [1, -1], [0, 2], [1, 1]),
        # Term 4: (a+b)(g)
        ([0, 1], [1, 1], [5], [1], [0, 1], [-1, 1]),
        # Term 5: (c-b)(e+f)
        ([2, 0], [1, -1], [0, 1], [1, 1], [2], [1]),
        # Term 6: (d-a)(g+h)
        ([1, 3], [1, -1], [5, 4], [1, 1], [0], [1]),
    ]
    
    # For the tensor power approach, we build the 4x4x4 decomposition (k=2)
    # and then project to 3x4x7.
    # 
    # The 4x4x4 grid uses indices 0-3 in each dimension.
    # We need: i in 0-2 (n=3), j in 0-3 (m=4), k in 0-6 (p=7)
    # But the 4x4x4 grid only has k in 0-3, so we can't directly use k=2 power for p=7.
    #
    # Alternative: Use k=3 power (8x8x8 grid, 343 terms) and project.
    # But 343 > 78, so that's worse.
    #
    # Better approach: The additive block decomposition already gives 78.
    # Let me try a different splitting strategy.
    #
    # Actually, let me reconsider. The current decomposition:
    # 3x4x7 = (2x4x7) + (1x4x7)
    # 2x4x7 = (2x2x7) + (2x2x7)
    # 2x2x7 = (2x2x2) + (2x2x2) + (2x2x2) + (2x2x1) = 7+7+7+4 = 25
    # So 2x4x7 = 50, 1x4x7 = 28, total = 78.
    #
    # Can we do better? What about 3x4x7 = (3x2x7) + (3x2x7)?
    # 3x2x7 = (2x2x7) + (1x2x7) = 25 + 14 = 39
    # Total = 78. Same.
    #
    # What about a more clever decomposition?
    # 3x4x7: Can we use a 3x3x3 block with a known decomposition?
    # The 3x3x3 matrix multiplication tensor has rank 23 (known result).
    # 3x4x7 = (3x3x7) + (3x1x7)
    # 3x3x7 = (3x3x3) + (3x3x3) + (3x3x1) = 23 + 23 + 9 = 55
    # 3x1x7 = 21
    # Total = 76. Better!
    #
    # But I need the actual rank-23 decomposition of 3x3x3.
    # Let me use a known one. The 3x3x3 tensor has rank 23.
    #
    # Actually, let me try yet another approach. What if I use:
    # 3x4x7 = (3x4x4) + (3x4x3)
    # 3x4x4 = (2x4x4) + (1x4x4) = (2x2x4 + 2x2x4) + 16 = (7+7+7+4)*2... no
    # 2x2x4 = (2x2x2) + (2x2x2) = 14
    # 2x4x4 = 2x2x4 + 2x2x4 = 28
    # 3x4x4 = 28 + 16 = 44
    # 3x4x3 = (2x4x3) + (1x4x3) = (2x2x3 + 2x2x3) + 12
    # 2x2x3 = (2x2x2) + (2x2x1) = 7 + 4 = 11
    # 2x4x3 = 22
    # 3x4x3 = 22 + 12 = 34
    # Total = 44 + 34 = 78. Same again.
    #
    # Let me try the 3x3x3 rank 23 approach more carefully.
    # I'll implement a known rank-23 decomposition for 3x3x3.
    
    # Rank-23 decomposition of 3x3x3 matrix multiplication
    # This is a well-known result. Let me use a specific decomposition.
    # Reference: B. Le Gall, "On the complexity of multiplication of 3x3 matrices"
    
    # I'll use a simpler approach: just optimize the block decomposition
    # by trying different splits and picking the best.
    
    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        u = [Fraction(0)] * (n * m)
        v = [Fraction(0)] * (m * p)
        w = [Fraction(0)] * (p * n)
        for idx, val in zip(u_indices, u_vals):
            u[idx] += Fraction(val)
        for idx, val in zip(v_indices, v_vals):
            v[idx] += Fraction(val)
        for idx, val in zip(w_indices, w_vals):
            w[idx] += Fraction(val)
        return (u, v, w)
    
    def gen_block(n_l, m_l, p_l, i_off, j_off, k_off):
        """Generate rank-1 terms for an n_l x m_l x p_l block."""
        terms = []
        
        if n_l == 1 and m_l == 1 and p_l == 1:
            i = i_off; j = j_off; k = k_off
            terms.append(add_rank1_term([i*m+j], [1], [j*p+k], [1], [k*n+i], [1]))
            return terms
        
        if n_l == 1:
            for j in range(m_l):
                for k in range(p_l):
                    jj = j_off + j; kk = k_off + k
                    terms.append(add_rank1_term([i_off*m+jj], [1], [jj*p+kk], [1], [kk*n+i_off], [1]))
            return terms
        
        if m_l == 1:
            for i in range(n_l):
                for k in range(p_l):
                    ii = i_off + i; kk = k_off + k
                    terms.append(add_rank1_term([ii*m+j_off], [1], [j_off*p+kk], [1], [kk*n+ii], [1]))
            return terms
        
        if p_l == 1:
            for i in range(n_l):
                for j in range(m_l):
                    ii = i_off + i; jj = j_off + j
                    terms.append(add_rank1_term([ii*m+jj], [1], [jj*p+k_off], [1], [k_off*n+ii], [1]))
            return terms
        
        if n_l == 2 and m_l == 2 and p_l == 2:
            i0 = i_off; i1 = i_off + 1
            j0 = j_off; j1 = j_off + 1
            k0 = k_off; k1 = k_off + 1
            for (ui, uv, vi, vv, wi, wv) in strassen_terms:
                u_idx = [i*m+j for i, j in zip([0,1,2,3], [0,0,1,1])]
                # Map Strassen indices to actual positions
                u_map = {0: i0*m+j0, 1: i0*m+j1, 2: i1*m+j0, 3: i1*m+j1}
                v_map = {0: j0*p+k0, 1: j0*p+k1, 2: j1*p+k0, 3: j1*p+k1, 4: j1*p+k0, 5: j1*p+k1}
                # Actually let me just use the explicit form
                pass
            
            # Use explicit Strassen terms
            s = [
                ([i0*m+j0, i1*m+j1], [1, 1],
                 [j0*p+k0, j1*p+k1], [1, 1],
                 [k0*n+i0, k1*n+i1], [1, 1]),
                ([i1*m+j0, i1*m+j1], [1, 1],
                 [j0*p+k0], [1],
                 [k0*n+i1, k1*n+i1], [1, -1]),
                ([i0*m+j0], [1],
                 [j0*p+k1, j1*p+k1], [1, -1],
                 [k1*n+i0, k1*n+i1], [1, 1]),
                ([i1*m+j1], [1],
                 [j1*p+k0, j0*p+k0], [1, -1],
                 [k0*n+i0, k0*n+i1], [1, 1]),
                ([i0*m+j0, i0*m+j1], [1, 1],
                 [j1*p+k1], [1],
                 [k0*n+i0, k1*n+i0], [-1, 1]),
                ([i1*m+j0, i0*m+j0], [1, -1],
                 [j0*p+k0, j0*p+k1], [1, 1],
                 [k1*n+i1], [1]),
                ([i0*m+j1, i1*m+j1], [1, -1],
                 [j1*p+k0, j1*p+k1], [1, 1],
                 [k0*n+i0], [1]),
            ]
            for (ui, uv, vi, vv, wi, wv) in s:
                terms.append(add_rank1_term(ui, uv, vi, vv, wi, wv))
            return terms
        
        # Recursive split: try splitting along the largest dimension
        # Split n if n_l >= 2
        if n_l >= 2:
            n1 = n_l // 2
            n2 = n_l - n1
            terms.extend(gen_block(n1, m_l, p_l, i_off, j_off, k_off))
            terms.extend(gen_block(n2, m_l, p_l, i_off + n1, j_off, k_off))
            return terms
        
        if m_l >= 2:
            m1 = m_l // 2
            m2 = m_l - m1
            terms.extend(gen_block(n_l, m1, p_l, i_off, j_off, k_off))
            terms.extend(gen_block(n_l, m2, p_l, i_off, j_off + m1, k_off))
            return terms
        
        if p_l >= 2:
            p1 = p_l // 2
            p2 = p_l - p1
            terms.extend(gen_block(n_l, m_l, p1, i_off, j_off, k_off))
            terms.extend(gen_block(n_l, m_l, p2, i_off, j_off, k_off + p1))
            return terms
        
        # Fallback schoolbook
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    ii = i_off + i; jj = j_off + j; kk = k_off + k
                    terms.append(add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1]))
        return terms
    
    # Try multiple splitting strategies and pick the best
    def count_terms(n_l, m_l, p_l):
        """Count rank-1 terms for a block using recursive splitting."""
        if n_l == 1 and m_l == 1 and p_l == 1:
            return 1
        if n_l == 1:
            return m_l * p_l
        if m_l == 1:
            return n_l * p_l
        if p_l == 1:
            return n_l * m_l
        if n_l == 2 and m_l == 2 and p_l == 2:
            return 7
        
        # Try all possible splits
        best = float('inf')
        # Split n
        for n1 in range(1, n_l):
            n2 = n_l - n1
            c = count_terms(n1, m_l, p_l) + count_terms(n2, m_l, p_l)
            if c < best:
                best = c
        # Split m
        for m1 in range(1, m_l):
            m2 = m_l - m1
            c = count_terms(n_l, m1, p_l) + count_terms(n_l, m2, p_l)
            if c < best:
                best = c
        # Split p
        for p1 in range(1, p_l):
            p2 = p_l - p1
            c = count_terms(n_l, m_l, p1) + count_terms(n_l, m_l, p2)
            if c < best:
                best = c
        
        # Also consider schoolbook
        schoolbook = n_l * m_l * p_l
        if schoolbook < best:
            best = schoolbook
        
        return best
    
    # Find optimal split
    optimal_rank = count_terms(n, m, p)
    
    # Now generate the actual decomposition using the optimal split
    def gen_optimal(n_l, m_l, p_l, i_off, j_off, k_off):
        """Generate terms using the optimal split strategy."""
        if n_l == 1 and m_l == 1 and p_l == 1:
            i = i_off; j = j_off; k = k_off
            return [add_rank1_term([i*m+j], [1], [j*p+k], [1], [k*n+i], [1])]
        
        if n_l == 1:
            terms = []
            for j in range(m_l):
                for k in range(p_l):
                    jj = j_off + j; kk = k_off + k
                    terms.append(add_rank1_term([i_off*m+jj], [1], [jj*p+kk], [1], [kk*n+i_off], [1]))
            return terms
        
        if m_l == 1:
            terms = []
            for i in range(n_l):
                for k in range(p_l):
                    ii = i_off + i; kk = k_off + k
                    terms.append(add_rank1_term([ii*m+j_off], [1], [j_off*p+kk], [1], [kk*n+ii], [1]))
            return terms
        
        if p_l == 1:
            terms = []
            for i in range(n_l):
                for j in range(m_l):
                    ii = i_off + i; jj = j_off + j
                    terms.append(add_rank1_term([ii*m+jj], [1], [jj*p+k_off], [1], [k_off*n+ii], [1]))
            return terms
        
        if n_l == 2 and m_l == 2 and p_l == 2:
            i0 = i_off; i1 = i_off + 1
            j0 = j_off; j1 = j_off + 1
            k0 = k_off; k1 = k_off + 1
            s = [
                ([i0*m+j0, i1*m+j1], [1, 1],
                 [j0*p+k0, j1*p+k1], [1, 1],
                 [k0*n+i0, k1*n+i1], [1, 1]),
                ([i1*m+j0, i1*m+j1], [1, 1],
                 [j0*p+k0], [1],
                 [k0*n+i1, k1*n+i1], [1, -1]),
                ([i0*m+j0], [1],
                 [j0*p+k1, j1*p+k1], [1, -1],
                 [k1*n+i0, k1*n+i1], [1, 1]),
                ([i1*m+j1], [1],
                 [j1*p+k0, j0*p+k0], [1, -1],
                 [k0*n+i0, k0*n+i1], [1, 1]),
                ([i0*m+j0, i0*m+j1], [1, 1],
                 [j1*p+k1], [1],
                 [k0*n+i0, k1*n+i0], [-1, 1]),
                ([i1*m+j0, i0*m+j0], [1, -1],
                 [j0*p+k0, j0*p+k1], [1, 1],
                 [k1*n+i1], [1]),
                ([i0*m+j1, i1*m+j1], [1, -1],
                 [j1*p+k0, j1*p+k1], [1, 1],
                 [k0*n+i0], [1]),
            ]
            return [add_rank1_term(ui, uv, vi, vv, wi, wv) for (ui, uv, vi, vv, wi, wv) in s]
        
        # Find optimal split
        best_cost = float('inf')
        best_split = None
        
        # Try splitting n
        for n1 in range(1, n_l):
            n2 = n_l - n1
            c = count_terms(n1, m_l, p_l) + count_terms(n2, m_l, p_l)
            if c < best_cost:
                best_cost = c
                best_split = ('n', n1, n2)
        
        # Try splitting m
        for m1 in range(1, m_l):
            m2 = m_l - m1
            c = count_terms(n_l, m1, p_l) + count_terms(n_l, m2, p_l)
            if c < best_cost:
                best_cost = c
                best_split = ('m', m1, m2)
        
        # Try splitting p
        for p1 in range(1, p_l):
            p2 = p_l - p1
            c = count_terms(n_l, m_l, p1) + count_terms(n_l, m_l, p2)
            if c < best_cost:
                best_cost = c
                best_split = ('p', p1, p2)
        
        # Schoolbook fallback
        schoolbook_cost = n_l * m_l * p_l
        if schoolbook_cost < best_cost:
            terms = []
            for i in range(n_l):
                for j in range(m_l):
                    for k in range(p_l):
                        ii = i_off + i; jj = j_off + j; kk = k_off + k
                        terms.append(add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1]))
            return terms
        
        terms = []
        dim, s1, s2 = best_split
        if dim == 'n':
            terms.extend(gen_optimal(s1, m_l, p_l, i_off, j_off, k_off))
            terms.extend(gen_optimal(s2, m_l, p_l, i_off + s1, j_off, k_off))
        elif dim == 'm':
            terms.extend(gen_optimal(n_l, s1, p_l, i_off, j_off, k_off))
            terms.extend(gen_optimal(n_l, s2, p_l, i_off, j_off + s1, k_off))
        else:
            terms.extend(gen_optimal(n_l, m_l, s1, i_off, j_off, k_off))
            terms.extend(gen_optimal(n_l, m_l, s2, i_off, j_off, k_off + s1))
        
        return terms
    
    all_terms = gen_optimal(n, m, p, 0, 0, 0)
    
    U_vecs = [t[0] for t in all_terms]
    V_vecs = [t[1] for t in all_terms]
    W_vecs = [t[2] for t in all_terms]
    
    # Verify
    valid = True
    for ci in range(n):
        for cj in range(m):
            for ck in range(p):
                prod_sum = sum(U_vecs[t][ci*m + cj] * V_vecs[t][cj*p + ck] * W_vecs[t][ck*n + ci] for t in range(len(U_vecs)))
                if prod_sum != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        U_vecs = []
        V_vecs = []
        W_vecs = []
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    term = add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1])
                    U_vecs.append(term[0])
                    V_vecs.append(term[1])
                    W_vecs.append(term[2])
    
    # Try to reduce rank by peeling
    if time.time() < DEADLINE - 10:
        U_vecs, V_vecs, W_vecs = peel_terms(U_vecs, V_vecs, W_vecs, n, m, p, DEADLINE - 10)
    
    # Final verification
    final_valid = True
    for ci in range(n):
        for cj in range(m):
            for ck in range(p):
                prod_sum = sum(U_vecs[t][ci*m + cj] * V_vecs[t][cj*p + ck] * W_vecs[t][ck*n + ci] for t in range(len(U_vecs)))
                if abs(float(prod_sum) - 1.0) > 1e-6:
                    final_valid = False
                    break
            if not final_valid:
                break
        if not final_valid:
            break
    
    if not final_valid:
        # Fallback to the original decomposition
        all_terms_orig = gen_optimal(n, m, p, 0, 0, 0)
        U_vecs = [t[0] for t in all_terms_orig]
        V_vecs = [t[1] for t in all_terms_orig]
        W_vecs = [t[2] for t in all_terms_orig]
    
    U = np.array([[float(x) for x in row] for row in U_vecs]).T
    V = np.array([[float(x) for x in row] for row in V_vecs]).T
    W = np.array([[float(x) for x in row] for row in W_vecs]).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
