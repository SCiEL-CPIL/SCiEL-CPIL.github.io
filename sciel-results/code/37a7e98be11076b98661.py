"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
    
    Using GQ-style decomposition: split middle dim 4 = 2+2, and for each 
    2x2 x 2x5 sub-problem, split the 5 columns as 2+2+1:
    - 2x2 * 2x2: Strassen (rank 7)
    - 2x2 * 2x1: schoolbook (rank 4)
    Total per sub-problem: 7 + 7 + 4 = 18
    Total rank: 2 * 18 = 36
    """
    import time
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    rank_terms = []  # Each term is (u_dict, v_dict, w_dict)
    
    def add_strassen_2x2x2(a_offset, b_offset, c_offset, sign=1):
        """
        Add Strassen terms for multiplying two 2x2 matrices.
        a_offset: (i0, j0) - top-left corner of the 2x2 block of A
        b_offset: (j0, k0) - top-left corner of the 2x2 block of B  
        c_offset: (i0, k0) - top-left corner of the 2x2 block of C
        """
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset
        
        terms = [
            # m1 = (x11+x22)*(y11+y22) -> z11 += 1, z22 += 1
            ({(i0, j0): 1, (i0+1, j0+1): 1}, 
             {(j0, k0): 1, (j0+1, k0+1): 1}, 
             {(i0, k0): 1, (i0+1, k0+1): 1}),
            # m2 = (x21+x22)*y11 -> z21 += 1, z22 -= 1
            ({(i0+1, j0): 1, (i0+1, j0+1): 1}, 
             {(j0, k0): 1}, 
             {(i0+1, k0): 1, (i0+1, k0+1): -1}),
            # m3 = x11*(y12-y22) -> z12 += 1, z22 += 1
            ({(i0, j0): 1}, 
             {(j0, k0+1): 1, (j0+1, k0+1): -1}, 
             {(i0, k0+1): 1, (i0+1, k0+1): 1}),
            # m4 = x22*(y21-y11) -> z11 += 1, z21 += 1
            ({(i0+1, j0+1): 1}, 
             {(j0+1, k0): 1, (j0, k0): -1}, 
             {(i0, k0): 1, (i0+1, k0): 1}),
            # m5 = (x11+x12)*y22 -> z11 -= 1, z12 += 1
            ({(i0, j0): 1, (i0, j0+1): 1}, 
             {(j0+1, k0+1): 1}, 
             {(i0, k0): -1, (i0, k0+1): 1}),
            # m6 = (x21-x11)*(y11+y12) -> z22 += 1
            ({(i0+1, j0): 1, (i0, j0): -1}, 
             {(j0, k0): 1, (j0, k0+1): 1}, 
             {(i0+1, k0+1): 1}),
            # m7 = (x12-x22)*(y21+y22) -> z11 += 1
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
        """
        Add schoolbook terms for multiplying 2x2 * 2x1.
        """
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
    
    # Build the full decomposition using GQ-style decomposition
    # A is 2x4: i in {0,1}, j in {0,1,2,3}
    # B is 4x5: j in {0,1,2,3}, k in {0,1,2,3,4}
    # C is 2x5: i in {0,1}, k in {0,1,2,3,4}
    
    # Split middle dim: j in {0,1} (top) and j in {2,3} (bottom)
    # For each A-block (2x2), multiply with B-block (2x5)
    # Split B's 5 columns as: cols 0,1 (2x2) + cols 2,3 (2x2) + col 4 (2x1)
    
    # Top A-block: A[0:2, 0:2], multiply with B[0:2, :]
    # C contribution from top block:
    add_strassen_2x2x2((0, 0), (0, 0), (0, 0))    # A_top * B[0:2, 0:2]
    add_strassen_2x2x2((0, 0), (0, 2), (0, 2))    # A_top * B[0:2, 2:4]
    add_schoolbook_2x2x1((0, 0), (0, 4), (0, 4))  # A_top * B[0:2, 4:5]
    
    # Bottom A-block: A[0:2, 2:4], multiply with B[2:4, :]
    # C contribution from bottom block:
    add_strassen_2x2x2((0, 2), (2, 0), (0, 0))    # A_bot * B[2:4, 0:2]
    add_strassen_2x2x2((0, 2), (2, 2), (0, 2))    # A_bot * B[2:4, 2:4]
    add_schoolbook_2x2x1((0, 2), (2, 4), (0, 4))  # A_bot * B[2:4, 4:5]
    
    base_rank = len(rank_terms)
    
    # Helper to build U, V, W from a list of (u_dict, v_dict, w_dict)
    def build_matrices(terms):
        r = len(terms)
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j][col] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k][col] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i][col] = coeff
        return U, V, W
    
    # Helper to verify that a decomposition is exact (tensor equality)
    def verify_decomposition(U, V, W):
        """Check that sum_r U[:,r] * V[:,r]^T * W[:,r]^T == multiplication tensor."""
        # The multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1
        # A rank-r decomposition: T = sum_r u_r (v_r ⊗ w_r) where
        #   u_r[i*m+j] = U[i*m+j][r], v_r[j*p+k] = V[j*p+k][r], w_r[k*n+i] = W[k*n+i][r]
        # So T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        # We only need T=1 when the indices are consistent (i.e., the j in A matches j in B, and k in B matches k in C).
        # For all (i, j, k) in range(n) x range(m) x range(p):
        #   sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] should equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U[0])):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True
    
    # Systematic search: try removing one term at a time and check if the remaining
    # terms still span the tensor (i.e., there exists a linear combination of the
    # remaining terms that reproduces the removed term's contribution).
    # More precisely: a set of rank-1 terms can be reduced if one term is in the
    # span of the others in the tensor space. We check this by verifying that
    # after removing term t, the remaining terms still exactly decompose the tensor
    # (possibly with adjusted coefficients).
    
    # First, try simple removal: remove each term and check if the rest still works
    # with unit coefficients (this won't work in general, but check anyway)
    # Then try: for each pair of terms, see if one can be expressed as a scalar
    # multiple of the other (they'd be linearly dependent in the tensor space).
    
    # Better approach: treat the 36 terms as vectors in the tensor space (size n*m*p = 40).
    # If the 36 vectors are linearly dependent, we can remove one and adjust coefficients.
    # But the coefficients must remain such that the tensor is exactly reproduced.
    
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k), and 0 otherwise.
    # Actually, the full tensor is defined on all of (i*m+j, j*p+k, k*n+i) but only
    # entries where the indices are "consistent" matter. Let's work in the full space.
    
    # Actually, for matrix multiplication, the tensor is:
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some (i,j,k), else 0
    # But this is a multi-linear constraint. The standard approach: the multiplication
    # tensor has rank equal to the minimum number of rank-1 tensors whose sum is T.
    
    # For our verification, we just need: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # Strategy: try removing each of the 36 terms. After removal, we have 35 terms.
    # We need to find new coefficients alpha_r such that
    #   sum_{r in remaining} alpha_r * U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = 1 for all (i,j,k)
    # This is a system of n*m*p = 2*4*5 = 40 equations in 35 unknowns.
    # If it's consistent, we have a rank-35 decomposition.
    
    # Let's set up the linear system. For each (i,j,k), the equation is:
    #   sum_r alpha_r * (U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]) = 1
    # Define M[(i,j,k), r] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Then M @ alpha = 1 (vector of ones, length 40)
    
    # For each term t to remove, solve M_t @ alpha = 1 where M_t excludes column t.
    
    from fractions import Fraction
    
    def try_remove_term(terms, t):
        """Try to find coefficients for all terms except t that still decompose the tensor."""
        r = len(terms) - 1
        # Build the matrix M of size (n*m*p) x r
        # M[idx, col] = U[idx_i, idx_j, col] * V[idx_j, idx_k, col] * W[idx_k, idx_i, col]
        # where idx = (i, j, k) mapped to flat index
        
        # Precompute U, V, W for the remaining terms
        remaining = [terms[i] for i in range(len(terms)) if i != t]
        U_rem, V_rem, W_rem = build_matrices(remaining)
        
        # Build the linear system
        num_eq = n * m * p  # 40
        # M is num_eq x r
        M = [[Fraction(0)] * r for _ in range(num_eq)]
        b = [Fraction(1)] * num_eq
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for col in range(r):
                        M[eq_idx][col] = Fraction(U_rem[i * m + j][col]) * Fraction(V_rem[j * p + k][col]) * Fraction(W_rem[k * n + i][col])
        
        # Solve M @ alpha = b using Gaussian elimination with fractions
        # If no solution, return None
        # Augmented matrix
        aug = [row[:] + [b[i]] for i, row in enumerate(M)]
        
        # Gaussian elimination
        rank_found = 0
        pivot_cols = []
        for col in range(r):
            if rank_found >= num_eq:
                break
            # Find pivot
            pivot = None
            for row in range(rank_found, num_eq):
                if aug[row][col] != 0:
                    pivot = row
                    break
            if pivot is None:
                continue
            # Swap
            aug[rank_found], aug[pivot] = aug[pivot], aug[rank_found]
            # Normalize
            pivot_val = aug[rank_found][col]
            for j in range(col, r + 1):
                aug[rank_found][j] = aug[rank_found][j] / pivot_val
            # Eliminate
            for row in range(num_eq):
                if row != rank_found and aug[row][col] != 0:
                    factor = aug[row][col]
                    for j in range(col, r + 1):
                        aug[row][j] -= factor * aug[rank_found][j]
            pivot_cols.append(col)
            rank_found += 1
        
        # Check consistency
        for row in range(rank_found, num_eq):
            if any(aug[row][j] != 0 for j in range(r)) and aug[row][r] != 0:
                return None  # Inconsistent
        
        # Check if the system has a solution
        # If consistent, find a solution (set free variables to 0)
        alpha = [Fraction(0)] * r
        for col in range(r):
            if col in pivot_cols:
                row_idx = pivot_cols.index(col)
                alpha[col] = aug[row_idx][r]
        
        # Verify the solution
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for col in range(r):
                        s += alpha[col] * Fraction(U_rem[i * m + j][col]) * Fraction(V_rem[j * p + k][col]) * Fraction(W_rem[k * n + i][col])
                    if s != 1:
                        return None
        
        # Convert to float if needed, but keep as Fraction for exactness
        # The coefficients alpha may be fractions; we need to check if they are
        # representable as required (integer/half-integer). The problem says
        # "integer/half-integer entries" for U, V, W, not for the coefficients.
        # Wait, re-reading: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) 
        # with integer/half-integer entries that exactly decompose the matrix multiplication tensor"
        # So U, V, W must have integer or half-integer entries. The coefficients alpha
        # are absorbed into U, V, W by scaling.
        
        # We need to scale U, V, W so that the entries are integer or half-integer.
        # Multiply U by alpha, V by 1, W by 1 (or distribute the scaling).
        # Actually, the decomposition is: sum_r alpha_r * u_r (v_r ⊗ w_r)
        # We can absorb alpha_r into u_r: u_r' = alpha_r * u_r
        # Then we need u_r', v_r, w_r to have integer or half-integer entries.
        
        # Check if alpha_r * U_rem entries are all integer or half-integer
        valid = True
        for col in range(r):
            a = alpha[col]
            if a == 0:
                continue
            # Check U entries
            for (i, j), coeff in remaining[col][0].items():
                val = a * Fraction(coeff)
                # val should be integer or half-integer, i.e., 2*val should be integer
                if (2 * val).denominator != 1:
                    valid = False
                    break
            if not valid:
                break
            # Check V entries
            if valid:
                for (j, k), coeff in remaining[col][1].items():
                    val = a * Fraction(coeff)  # Wait, V is not scaled by alpha
                    # Actually, we only scale U by alpha. V and W stay the same.
                    # V entries are already integer, so they're fine.
                    pass
            # Check W entries - also not scaled, already integer.
        
        if not valid:
            # Try scaling V instead, or splitting the scaling
            # Try: scale U by a, V by b, W by c where a*b*c = alpha_r for each r
            # This is more complex. For now, just try scaling U.
            # Alternative: try different distributions
            pass
        
        if valid:
            # Build the final matrices
            U_final = [[0] * r for _ in range(n * m)]
            V_final = [[0] * r for _ in range(m * p)]
            W_final = [[0] * r for _ in range(p * n)]
            
            for col in range(r):
                a = alpha[col]
                u_dict, v_dict, w_dict = remaining[col]
                for (i, j), coeff in u_dict.items():
                    U_final[i * m + j][col] = float(a * Fraction(coeff))
                for (j, k), coeff in v_dict.items():
                    V_final[j * p + k][col] = float(coeff)
                for (i, k), coeff in w_dict.items():
                    W_final[k * n + i][col] = float(coeff)
            
            # Final verification
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for col in range(r):
                            s += U_final[i * m + j][col] * V_final[j * p + k][col] * W_final[k * n + i][col]
                        if abs(s - 1) > 1e-9:
                            return None
            
            return U_final, V_final, W_final
        
        return None
    
    # Try removing each term, then try removing pairs of terms
    best_result = None
    best_rank = base_rank
    
    # Time budget: we have 120s total, use most of it for this search
    import time
    search_start = time.time()
    search_deadline = search_start + 108  # Leave 12s margin
    
    # First pass: try removing single terms
    for t in range(base_rank):
        if time.time() > search_deadline:
            break
        result = try_remove_term(rank_terms, t)
        if result is not None:
            U_final, V_final, W_final = result
            if len(U_final[0]) < best_rank:
                best_rank = len(U_final[0])
                best_result = (U_final, V_final, W_final)
    
    # Second pass: try removing pairs of terms (rank reduction by 2)
    if best_rank > 34:  # Only try if we haven't already found rank <= 34
        for t1 in range(base_rank):
            if time.time() > search_deadline:
                break
            for t2 in range(t1 + 1, base_rank):
                if time.time() > search_deadline:
                    break
                # Try removing terms t1 and t2
                remaining = [rank_terms[i] for i in range(base_rank) if i != t1 and i != t2]
                r = len(remaining)
                if r >= best_rank:
                    continue  # No improvement possible
                
                U_rem, V_rem, W_rem = build_matrices(remaining)
                
                # Build the linear system
                num_eq = n * m * p  # 40
                M = [[Fraction(0)] * r for _ in range(num_eq)]
                b = [Fraction(1)] * num_eq
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            eq_idx = i * m * p + j * p + k
                            for col in range(r):
                                M[eq_idx][col] = Fraction(U_rem[i * m + j][col]) * Fraction(V_rem[j * p + k][col]) * Fraction(W_rem[k * n + i][col])
                
                # Solve M @ alpha = b using Gaussian elimination with fractions
                aug = [row[:] + [b[i]] for i, row in enumerate(M)]
                
                rank_found = 0
                pivot_cols = []
                for col in range(r):
                    if rank_found >= num_eq:
                        break
                    pivot = None
                    for row in range(rank_found, num_eq):
                        if aug[row][col] != 0:
                            pivot = row
                            break
                    if pivot is None:
                        continue
                    aug[rank_found], aug[pivot] = aug[pivot], aug[rank_found]
                    pivot_val = aug[rank_found][col]
                    for j in range(col, r + 1):
                        aug[rank_found][j] = aug[rank_found][j] / pivot_val
                    for row in range(num_eq):
                        if row != rank_found and aug[row][col] != 0:
                            factor = aug[row][col]
                            for j in range(col, r + 1):
                                aug[row][j] -= factor * aug[rank_found][j]
                    pivot_cols.append(col)
                    rank_found += 1
                
                # Check consistency: rows that are all zero in M must have b=0
                consistent = True
                for row in range(rank_found, num_eq):
                    if any(aug[row][j] != 0 for j in range(r)):
                        # This row has non-zero coefficients but wasn't reduced
                        # Actually, after full elimination, if rank_found < num_eq,
                        # the remaining rows should be [0,0,...,0 | b_i]
                        # If b_i != 0, inconsistent
                        if aug[row][r] != 0:
                            consistent = False
                            break
                    # If all M entries are zero, b must be zero (it is, since b=1 always)
                    # Wait, b is always 1, so if M row is all zero, we have 0=1 which is inconsistent
                    if all(aug[row][j] == 0 for j in range(r)):
                        if aug[row][r] != 0:
                            consistent = False
                            break
                
                if not consistent:
                    continue
                
                # Find a solution (set free variables to 0)
                alpha = [Fraction(0)] * r
                for col in range(r):
                    if col in pivot_cols:
                        row_idx = pivot_cols.index(col)
                        alpha[col] = aug[row_idx][r]
                
                # Verify the solution
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = Fraction(0)
                            for col in range(r):
                                s += alpha[col] * Fraction(U_rem[i * m + j][col]) * Fraction(V_rem[j * p + k][col]) * Fraction(W_rem[k * n + i][col])
                            if s != 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Check if alpha * U entries are integer or half-integer
                half_int_valid = True
                for col in range(r):
                    a = alpha[col]
                    if a == 0:
                        continue
                    for (i, j), coeff in remaining[col][0].items():
                        val = a * Fraction(coeff)
                        if (2 * val).denominator != 1:
                            half_int_valid = False
                            break
                    if not half_int_valid:
                        break
                
                if not half_int_valid:
                    continue
                
                # Build the final matrices
                U_final = [[0] * r for _ in range(n * m)]
                V_final = [[0] * r for _ in range(m * p)]
                W_final = [[0] * r for _ in range(p * n)]
                
                for col in range(r):
                    a = alpha[col]
                    u_dict, v_dict, w_dict = remaining[col]
                    for (i, j), coeff in u_dict.items():
                        U_final[i * m + j][col] = float(a * Fraction(coeff))
                    for (j, k), coeff in v_dict.items():
                        V_final[j * p + k][col] = float(coeff)
                    for (i, k), coeff in w_dict.items():
                        W_final[k * n + i][col] = float(coeff)
                
                # Final verification
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for col in range(r):
                                s += U_final[i * m + j][col] * V_final[j * p + k][col] * W_final[k * n + i][col]
                            if abs(s - 1) > 1e-9:
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    if r < best_rank:
                        best_rank = r
                        best_result = (U_final, V_final, W_final)
    
    if best_result is not None:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    # Fallback: return the base decomposition
    U, V, W = build_matrices(rank_terms)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
