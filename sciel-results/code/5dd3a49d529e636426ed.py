"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    # Rank 17 decomposition for 2x3x3 matrix multiplication
    # Constructed by combining:
    # 1. Strassen (rank 7) for 2x2x2 part
    # 2. Schoolbook (rank 4) for 2x1x2 part (middle column)
    # 3. Schoolbook (rank 6) for 2x3x1 part (last column)
    
    # Total rank: 7 + 4 + 6 = 17
    rank = 17
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Indices for global tensors
    # A: i in 0..1, j in 0..2 -> idx = i*3 + j
    # B: j in 0..2, k in 0..2 -> idx = j*3 + k
    # C: k in 0..2, i in 0..1 -> idx = k*2 + i (Trace convention W[k*n+i, r])
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: 0,1,3,4
    # B indices: 0,1,3,4
    # C indices: 0,1,2,3
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10, -C11
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: A00(B01-B11) -> C01, C11
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: A11(B10-B00) -> C00, C10
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: (A00+A01)B11 -> -C00, C01
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    # A indices: 2, 5
    # B indices: 6, 7
    # C indices: 0,1,2,3
    
    # r=7: A02 B20 -> C00
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    # r=8: A02 B21 -> C01
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    # r=9: A12 B20 -> C10
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    # r=10: A12 B21 -> C11
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    # A indices: 0,1,2,3,4,5
    # B indices: 2,5,8
    # C indices: 4,5
    
    # r=11: A00 B02 -> C02
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    # r=12: A01 B12 -> C02
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    # r=13: A02 B22 -> C02
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    # r=14: A10 B02 -> C12
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    # r=15: A11 B12 -> C12
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    # r=16: A12 B22 -> C12
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    # Verify exact decomposition
    def _verify(U_mat, V_mat, W_mat, n_, m_, p_, rank_):
        for i in range(n_):
            for k in range(p_):
                for j1 in range(m_):
                    for j2 in range(m_):
                        s = 0.0
                        for r in range(rank_):
                            s += W_mat[k*n_+i, r] * U_mat[i*m_+j1, r] * V_mat[j2*p_+k, r]
                        target = 1.0 if j1 == j2 else 0.0
                        if abs(s - target) > 1e-9:
                            return False
        return True
    
    if not _verify(U, V, W, n, m, p, rank):
        # Fallback to schoolbook if verification fails (should not happen)
        rank_sb = n * m * p
        U_fb = np.zeros((n * m, rank_sb))
        V_fb = np.zeros((m * p, rank_sb))
        W_fb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Try to find rank 16 via structured linear-algebraic search
    # Strategy: Formulate the tensor decomposition as a system of linear equations
    # where each rank-16 term is a "merged" combination of the 17 basis terms.
    # We search over which pairs of the 17 terms to merge, then solve for
    # half-integer coefficients.
    if time.time() < deadline - 5.0:
        rank_test = 16
        found = False
        
        # Build the 17-term basis as column vectors in the tensor space
        # Each basis term t has a (i,j,k) pattern; we encode it as a vector
        # in the 18-dimensional space of (i,j1,k,j2) constraints.
        
        # The 17 basis terms from the rank-17 decomposition above:
        # Each term is defined by its U, V, W nonzero patterns
        basis_terms = []
        
        # Term 0: Strassen M1
        basis_terms.append({
            'U': [(0, 1), (4, 1)],
            'V': [(0, 1), (4, 1)],
            'W': [(0, 1), (3, 1)]
        })
        # Term 1: Strassen M2
        basis_terms.append({
            'U': [(3, 1), (4, 1)],
            'V': [(0, 1)],
            'W': [(1, 1), (3, -1)]
        })
        # Term 2: Strassen M3
        basis_terms.append({
            'U': [(0, 1)],
            'V': [(1, 1), (4, -1)],
            'W': [(2, 1), (3, 1)]
        })
        # Term 3: Strassen M4
        basis_terms.append({
            'U': [(4, 1)],
            'V': [(3, 1), (0, -1)],
            'W': [(0, 1), (1, 1)]
        })
        # Term 4: Strassen M5
        basis_terms.append({
            'U': [(0, 1), (1, 1)],
            'V': [(4, 1)],
            'W': [(0, -1), (2, 1)]
        })
        # Term 5: Strassen M6
        basis_terms.append({
            'U': [(3, 1), (0, -1)],
            'V': [(0, 1), (1, 1)],
            'W': [(3, 1)]
        })
        # Term 6: Strassen M7
        basis_terms.append({
            'U': [(1, 1), (4, -1)],
            'V': [(3, 1), (4, 1)],
            'W': [(0, 1)]
        })
        # Term 7: Schoolbook A02*B20 -> C00
        basis_terms.append({
            'U': [(2, 1)],
            'V': [(6, 1)],
            'W': [(0, 1)]
        })
        # Term 8: Schoolbook A02*B21 -> C01
        basis_terms.append({
            'U': [(2, 1)],
            'V': [(7, 1)],
            'W': [(2, 1)]
        })
        # Term 9: Schoolbook A12*B20 -> C10
        basis_terms.append({
            'U': [(5, 1)],
            'V': [(6, 1)],
            'W': [(1, 1)]
        })
        # Term 10: Schoolbook A12*B21 -> C11
        basis_terms.append({
            'U': [(5, 1)],
            'V': [(7, 1)],
            'W': [(3, 1)]
        })
        # Term 11: Schoolbook A00*B02 -> C02
        basis_terms.append({
            'U': [(0, 1)],
            'V': [(2, 1)],
            'W': [(4, 1)]
        })
        # Term 12: Schoolbook A01*B12 -> C02
        basis_terms.append({
            'U': [(1, 1)],
            'V': [(5, 1)],
            'W': [(4, 1)]
        })
        # Term 13: Schoolbook A02*B22 -> C02
        basis_terms.append({
            'U': [(2, 1)],
            'V': [(8, 1)],
            'W': [(4, 1)]
        })
        # Term 14: Schoolbook A10*B02 -> C12
        basis_terms.append({
            'U': [(3, 1)],
            'V': [(2, 1)],
            'W': [(5, 1)]
        })
        # Term 15: Schoolbook A11*B12 -> C12
        basis_terms.append({
            'U': [(4, 1)],
            'V': [(5, 1)],
            'W': [(5, 1)]
        })
        # Term 16: Schoolbook A12*B22 -> C12
        basis_terms.append({
            'U': [(5, 1)],
            'V': [(8, 1)],
            'W': [(5, 1)]
        })
        
        # For each pair (a, b) of the 17 terms, try to find a rank-16 decomposition
        # where one term is the sum of terms a and b (with half-integer coefficients)
        # and the other 15 terms remain unchanged.
        # Actually, more generally: we can replace any subset of the 17 terms with
        # fewer terms. The simplest case: merge two terms into one.
        
        # For a merged term to work, we need:
        # W_merged[i] * U_merged[j] * V_merged[k] = W_a[i]*U_a[j]*V_a[k] + W_b[i]*U_b[j]*V_b[k]
        # for all (i,j,k). This is a rank-1 approximation of the sum of two rank-1 tensors.
        
        # The sum of two rank-1 tensors is rank 1 iff one is a scalar multiple of the other.
        # So we check: is term a proportional to term b?
        
        from itertools import combinations
        
        for (a, b) in combinations(range(17), 2):
            if time.time() > deadline - 3.0:
                break
            
            # Check if term a and term b are proportional
            # Extract the (i,j,k) -> coefficient maps
            def get_coeff_map(term):
                """Get the 3D tensor coefficients for a basis term."""
                # The term is defined by U, V, W nonzeros
                # The 3D tensor T[i,j,k] = sum over the term's contribution
                # For a single term, T[i,j,k] = U[i] * V[j] * W[k]
                # where U[i], V[j], W[k] are the values at those indices (0 if not listed)
                
                U_map = dict(term['U'])
                V_map = dict(term['V'])
                W_map = dict(term['W'])
                
                return U_map, V_map, W_map
            
            Ua_map, Va_map, Wa_map = get_coeff_map(basis_terms[a])
            Ub_map, Vb_map, Wb_map = get_coeff_map(basis_terms[b])
            
            # Check proportionality: there exists lambda such that
            # Ua[i] = lambda * Ub[i] for all i
            # Va[j] = lambda * Vb[j] for all j
            # Wa[k] = lambda * Wb[k] for all k
            
            # Find lambda from U
            lambdas = []
            for i in set(list(Ua_map.keys()) + list(Ub_map.keys())):
                ua = Ua_map.get(i, 0)
                ub = Ub_map.get(i, 0)
                if ua != 0 and ub != 0:
                    lambdas.append(ua / ub)
                elif ua != 0 or ub != 0:
                    lambdas = None
                    break
            
            if lambdas is None or len(lambdas) == 0:
                continue
            
            lambda_val = lambdas[0]
            consistent = all(abs(l - lambda_val) < 1e-9 for l in lambdas)
            
            # Verify with V
            if consistent:
                for j in set(list(Va_map.keys()) + list(Vb_map.keys())):
                    va = Va_map.get(j, 0)
                    vb = Vb_map.get(j, 0)
                    if va != 0 and vb != 0:
                        if abs(va / vb - lambda_val) > 1e-9:
                            consistent = False
                            break
                    elif va != 0 or vb != 0:
                        consistent = False
                        break
            
            # Verify with W
            if consistent:
                for k in set(list(Wa_map.keys()) + list(Wb_map.keys())):
                    wa = Wa_map.get(k, 0)
                    wb = Wb_map.get(k, 0)
                    if wa != 0 and wb != 0:
                        if abs(wa / wb - lambda_val) > 1e-9:
                            consistent = False
                            break
                    elif wa != 0 or wb != 0:
                        consistent = False
                        break
            
            if not consistent:
                continue
            
            # Terms a and b are proportional! We can merge them.
            # The merged term has coefficients: (1+lambda) * term_b (or term_a)
            # Build the rank-16 decomposition
            merge_factor = 1.0 + lambda_val
            
            U_new = np.zeros((n*m, rank_test))
            V_new = np.zeros((m*p, rank_test))
            W_new = np.zeros((p*n, rank_test))
            
            col = 0
            for t in range(17):
                if t == a or t == b:
                    continue
                term = basis_terms[t]
                for idx, val in term['U']:
                    U_new[idx, col] = val
                for idx, val in term['V']:
                    V_new[idx, col] = val
                for idx, val in term['W']:
                    W_new[idx, col] = val
                col += 1
            
            # Add the merged term
            term_b = basis_terms[b]
            for idx, val in term_b['U']:
                U_new[idx, col] = val * merge_factor
            for idx, val in term_b['V']:
                V_new[idx, col] = val * merge_factor
            for idx, val in term_b['W']:
                W_new[idx, col] = val * merge_factor
            
            # Verify
            valid = _verify(U_new, V_new, W_new, n, m, p, rank_test)
            if valid:
                found = True
                U = U_new
                V = V_new
                W = W_new
                rank = rank_test
                break
        
        # If no proportional pair found, try a more general approach:
        # Look for a linear dependency among all 17 terms
        if not found and time.time() < deadline - 3.0:
            # Build the 17x18 matrix of tensor coefficients
            # Each term is a vector in the 18D space of (i,j1,k,j2) with i!=j1 or j1==j2
            # Actually, the tensor has 2*3*3*3 = 54 entries, but by symmetry only 18 are independent
            # (i, j1, k, j2) with the constraint that the tensor is nonzero only when j1==j2 for the output
            
            # Let's build the full 54D tensor for each term
            n_tensor = n * m * p * m  # 2*3*3*3 = 54
            
            def term_to_vector(term):
                """Convert a basis term to a 54D vector."""
                vec = np.zeros(n_tensor)
                U_map = dict(term['U'])
                V_map = dict(term['V'])
                W_map = dict(term['W'])
                
                for i in range(n):
                    for j1 in range(m):
                        for k in range(p):
                            for j2 in range(m):
                                u_val = U_map.get(i * m + j1, 0)
                                v_val = V_map.get(j2 * p + k, 0)
                                w_val = W_map.get(k * n + i, 0)
                                if u_val != 0 and v_val != 0 and w_val != 0:
                                    idx = ((i * m + j1) * p + k) * m + j2
                                    vec[idx] = u_val * v_val * w_val
                return vec
            
            # Build the 17 x 54 matrix
            M = np.zeros((17, n_tensor))
            for t in range(17):
                M[t, :] = term_to_vector(basis_terms[t])
            
            # Find the null space of M^T (i.e., linear dependencies among the 17 terms)
            # A dependency is a vector c in R^17 such that M^T c = 0
            # This means sum(c[t] * term_t) = 0 as a tensor
            
            # Use SVD to find the null space
            try:
                U_svd, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
                # The null space dimension is 17 - rank(M)
                tol = 1e-10
                rank_M = np.sum(s_vals > tol)
                null_dim = 17 - rank_M
                
                if null_dim > 0:
                    # The null space vectors are the last (17 - rank_M) rows of Vt
                    null_vecs = Vt[rank_M:, :]
                    
                    for nv in null_vecs:
                        if time.time() > deadline - 3.0:
                            break
                        
                        # nv is a 17D vector: sum(nv[t] * term_t) = 0
                        # This means we can replace the 17 terms with 16 terms by
                        # expressing one term as a combination of the others.
                        
                        # Find the term with the largest |nv[t]| to eliminate
                        max_idx = np.argmax(np.abs(nv))
                        max_val = nv[max_idx]
                        
                        if abs(max_val) < 1e-10:
                            continue
                        
                        # Express term max_idx as a combination of the others
                        # term_max = -sum_{t != max_idx} (nv[t] / nv[max_idx]) * term_t
                        
                        # Build the rank-16 decomposition
                        U_new = np.zeros((n*m, rank_test))
                        V_new = np.zeros((m*p, rank_test))
                        W_new = np.zeros((p*n, rank_test))
                        
                        col = 0
                        for t in range(17):
                            if t == max_idx:
                                continue
                            
                            coeff = -nv[t] / max_val
                            term = basis_terms[t]
                            
                            for idx, val in term['U']:
                                U_new[idx, col] = val * coeff
                            for idx, val in term['V']:
                                V_new[idx, col] = val * coeff
                            for idx, val in term['W']:
                                W_new[idx, col] = val * coeff
                            col += 1
                        
                        # Verify
                        valid = _verify(U_new, V_new, W_new, n, m, p, rank_test)
                        if valid:
                            found = True
                            U = U_new
                            V = V_new
                            W = W_new
                            rank = rank_test
                            break
            except Exception:
                pass
    
    # Final verification before returning
    if rank == 16:
        if not _verify(U, V, W, n, m, p, rank):
            # Revert to rank 17
            U = np.zeros((n * m, 17))
            V = np.zeros((m * p, 17))
            W = np.zeros((p * n, 17))
            U[0, 0] = 1; U[4, 0] = 1; V[0, 0] = 1; V[4, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
            U[3, 1] = 1; U[4, 1] = 1; V[0, 1] = 1; W[1, 1] = 1; W[3, 1] = -1
            U[0, 2] = 1; V[1, 2] = 1; V[4, 2] = -1; W[2, 2] = 1; W[3, 2] = 1
            U[4, 3] = 1; V[3, 3] = 1; V[0, 3] = -1; W[0, 3] = 1; W[1, 3] = 1
            U[0, 4] = 1; U[1, 4] = 1; V[4, 4] = 1; W[0, 4] = -1; W[2, 4] = 1
            U[3, 5] = 1; U[0, 5] = -1; V[0, 5] = 1; V[1, 5] = 1; W[3, 5] = 1
            U[1, 6] = 1; U[4, 6] = -1; V[3, 6] = 1; V[4, 6] = 1; W[0, 6] = 1
            U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
            U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
            U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
            U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
            U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
            U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
            U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1
            U[3, 14] = 1; V[2, 14] = 1; W[5, 14] = 1
            U[4, 15] = 1; V[5, 15] = 1; W[5, 15] = 1
            U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
            rank = 17
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
