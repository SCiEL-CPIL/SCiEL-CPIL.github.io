"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Kronecker canonical form approach for 2x4 * 4x5 matrix multiplication.
    Exploits n=2 structure: C = A*B where C is 2x5, A is 2x4, B is 4x5.
    The tensor T[i,j,k] = 1 for all (i,j,k) in [0,2)x[0,4)x[0,5).
    
    We view this as: for each output position (i,k), C[i,k] = sum_j A[i,j]*B[j,k].
    The tensor decomposes as sum_r U_r ⊗ V_r ⊗ W_r where
    U_r[i,j] * V_r[j,k] * W_r[i,k] sums to 1 for all (i,j,k).
    
    Using the n=2 structure, we can write the 2x5 output as two 1x5 vectors.
    The key insight: the multiplication tensor for 2x4*4x5 has rank at most 2*4*5/2 = 20
    if we can exploit the structure, but realistically we use block methods.
    
    Strategy: Use the fact that for n=2, we can separate the two output rows.
    Row 0: C[0,k] = sum_j A[0,j]*B[j,k]
    Row 1: C[1,k] = sum_j A[1,j]*B[j,k]
    
    Apply Winograd/FFT-style methods on the j dimension (size 4) to reduce multiplications.
    For each output column k (5 values), we need 2 dot products of length 4.
    Total: 2*5 = 10 dot products of length 4.
    
    Using 4-point FFT/Winograd: each dot product of length 4 can be done in 4 mults (schoolbook)
    or potentially fewer with structured methods.
    
    However, the tensor rank is what matters. Let's use the known result that
    R(2x4x5) can be approached via:
    - Split into 2x2x5 blocks: two 2x2x5 multiplications
    - Each 2x2x5 can be done with rank ~ 2*5*7/4 ≈ 17.5, so ~18 each = 36 total
    - Or use the full structure more carefully.
    
    Let me try a direct construction using the n=2 symmetry.
    """
    import time
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 2, 4, 5

    # For 2x4*4x5, the tensor has 2*4*5 = 40 entries, all equal to 1.
    # We seek a rank-r decomposition.
    
    # Approach: Use the structure where we split the middle dimension (j=0..3)
    # into pairs and use Strassen-like additions.
    
    # Key insight for n=2: The two rows of A interact with B independently
    # in terms of output, but share the same B columns.
    
    # Let's use a systematic approach:
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # 
    # We can write this using 4-point Winograd for the j-sum.
    # 4-point Winograd: x*y where x,y are length-4 vectors.
    # m0 = x0+x1, m1 = x0-x1, m2 = x2+x3, m3 = x2-x3
    # n0 = y0+y1, n1 = y0-y1, n2 = y2+y3, n3 = y2-y3
    # Then:
    # r0 = m0*n0 + m1*n1  (no wait, that's not right for dot product)
    # 
    # Actually for dot product sum_j x_j * y_j:
    # Standard: 4 multiplications
    # Winograd 4-point: 
    #   a = x0+x1, b = x0-x1, c = x2+x3, d = x2-x3
    #   e = y0+y1, f = y0-y1, g = y2+y3, h = y2-y3
    #   r = a*e + b*f + c*g + d*h  -- this is still 4 mults
    # 
    # The Winograd trick for convolution, not direct dot product.
    # For direct dot product of length 4, minimum is 4 (it's a bilinear form of rank 4).
    
    # So the real structure is: we have 10 independent dot products (2 rows * 5 cols).
    # But they share the B matrix columns.
    
    # Let me think differently. The tensor T[i,j,k] = 1 for all i,j,k.
    # This is the "all-ones" tensor of shape (2,4,5).
    # 
    # The rank of the all-ones tensor of shape (a,b,c) is known to be min(a,b,c) 
    # if we're talking about the standard tensor rank... no that's not right either.
    # 
    # Actually, the all-ones tensor T[i,j,k] = 1 has rank 1! Because
    # T = u ⊗ v ⊗ w where u[i]=1, v[j]=1, w[k]=1.
    # 
    # Wait, that can't be right for matrix multiplication. Let me re-examine.
    # 
    # For matrix multiplication C = AB, the tensor is NOT all ones.
    # The multiplication tensor for (n,m,p) matrices has T[i,j,k] = 1 if the
    # "position" (i,j,k) corresponds to a valid term in the product.
    # 
    # Actually, for standard matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor M[i,j,k] = 1 for all valid (i,j,k).
    # Yes! The multiplication tensor IS the all-ones tensor of shape (n,m,p).
    # 
    # But wait, that would mean rank 1, which is clearly wrong.
    # 
    # I'm confusing myself. Let me be precise.
    # 
    # The matrix multiplication tensor is defined as:
    # M = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * M_{ijk}
    # where M_{ijk} = 1 for all i,j,k.
    # 
    # No wait. The multiplication map is:
    # (A,B) -> C where C_{ik} = sum_j A_{ij} B_{jk}
    # 
    # This is a trilinear map. The associated tensor has:
    # T_{ijk} = 1 if we interpret it as the coefficient of A_{ij} B_{jk} in C_{ik}
    # 
    # But C_{ik} depends on ALL j, so T_{ijk} = 1 for all (i,j,k).
    # 
    # Hmm, but then the tensor would be rank 1 (outer product of three all-ones vectors).
    # That contradicts the known result that matrix multiplication has rank > np.
    # 
    # I think I'm making an error. Let me look at this more carefully.
    # 
    # The multiplication tensor for m x n * n x p matrices is:
    # M = sum_{i=1}^m sum_{j=1}^n sum_{k=1}^p E_{ij} ⊗ E_{jk} ⊗ E_{ik}
    # 
    # Wait no. The tensor is:
    # M = sum_{i,j,k} (E_{ij} ⊗ E_{jk}) ⊗ E_{ik} * delta_{valid}
    # 
    # Actually, the correct formulation:
    # The bilinear map (A,B) -> AB can be written as:
    # AB = sum_{j=1}^n (A e_j) (e_j^T B) = sum_j sum_i sum_k A_{ij} B_{jk} E_{ik}
    # 
    # So the tensor is:
    # M = sum_{i,j,k} E_{ij} ⊗ E_{jk} ⊗ E_{ik}
    # 
    # where the sum is over all i in [0,n), j in [0,m), k in [0,p).
    # 
    # So yes, M_{ijk} = 1 for all (i,j,k). The tensor IS the all-ones tensor.
    # 
    # But the RANK of this tensor is NOT 1! Because the rank of a tensor is the
    # minimum number of rank-1 tensors needed to sum to it, and the all-ones tensor
    # of shape (a,b,c) has rank... 
    # 
    # Actually, the all-ones tensor 1_a ⊗ 1_b ⊗ 1_c IS rank 1!
    # 
    # So I'm clearly misunderstanding something fundamental about the matrix
    # multiplication tensor. Let me reconsider.
    # 
    # OH WAIT. I think the issue is that the matrix multiplication tensor is NOT
    # simply the all-ones tensor. The correct tensor for matrix multiplication
    # (n x m) * (m x p) -> (n x p) is:
    # 
    # M = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} 
    #       (e_i ⊗ e_j) ⊗ (e_i ⊗ e_k) ... no this doesn't make sense dimensionally.
    # 
    # Let me use the standard definition. The multiplication tensor for
    # Mat(n,m) * Mat(m,p) -> Mat(n,p) is an element of:
    # Mat(n,m)^* ⊗ Mat(m,p)^* ⊗ Mat(n,p)
    # 
    # In coordinates: M = sum_{i,j,k,l} M_{ijkl} E^i_j ⊗ E^j_k ⊗ E_{kl}
    # where M_{ijkl} = 1 if i=k... no.
    # 
    # C_{kl} = sum_j A_{ij} B_{jk}  -- wait, I'm mixing up indices.
    # 
    # Let me use: C_{ik} = sum_j A_{ij} B_{jk}
    # 
    # The tensor is: M = sum_{i,j,k} E_{ij} ⊗ E_{jk} ⊗ E_{ik}
    # 
    # This IS the all-ones tensor of shape (n*m, m*p, n*p) with specific structure.
    # No wait, the shapes are (n*m) x (m*p) x (n*p) for the flattened versions.
    # 
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (n*p, rank)
    # 
    # And we need: sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
    # where T[i*m+j, j'*p+k, k'*n+i'] = 1 if j=j' and i=i'... 
    # 
    # NO. The tensor T has T[a,b,c] = 1 for all a,b,c in their respective ranges
    # IF AND ONLY IF the indices are "compatible" in the multiplication sense.
    # 
    # Specifically: T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k).
    # And T[a,b,c] = 0 otherwise.
    # 
    # So it's NOT the all-ones tensor! It's a structured tensor where only
    # specific (a,b,c) triples have value 1.
    # 
    # For (a,b,c) = (i*m+j, j*p+k, k*n+i) for some (i,j,k), T[a,b,c] = 1.
    # For other (a,b,c), T[a,b,c] = 0.
    # 
    # This makes much more sense! The tensor has a lot of zeros.
    # 
    # So for our problem: n=2, m=4, p=5
    # U: 8 x rank
    # V: 20 x rank  
    # W: 10 x rank
    # 
    # T[i*4+j, j*5+k, k*2+i] = 1 for i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4}
    # T[a,b,c] = 0 otherwise.
    # 
    # Number of non-zero entries: 2*4*5 = 40.
    # 
    # Now, the rank of this tensor is what we need to minimize.
    
    # Let me construct a decomposition using the block-Strassen approach
    # but optimized for the n=2 case.
    
    # The current best is rank 36. Let me see if I can do better.
    
    # For 2x4*4x5, one approach:
    # Split into 2x2*2x5 and 2x2*2x5 blocks (splitting j into {0,1} and {2,3})
    # Each 2x2*2x5 has tensor of shape (4, 10, 10) with 2*2*5=20 non-zeros.
    # 
    # For 2x2*2x5: C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k]
    # This is just 2x2*2x5 matrix multiplication.
    # 
    # The rank of 2x2*2x5 multiplication: 
    # We can write it as: for each k, C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # and C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    # 
    # Using Strassen for the 2x2 part: 7 multiplications per "column pair"
    # But here we have 5 columns, so we need to be more careful.
    # 
    # Actually, 2x2*2x5 can be seen as: multiply 2x2 by 2x5.
    # This is equivalent to 5 independent 2x2*2x1 multiplications sharing the 2x2 matrix.
    # 
    # For 2x2*2x1: rank 7 (Strassen) or rank 4 (schoolbook)
    # With 5 columns sharing the A matrix, we can potentially do better.
    # 
    # Let me use a different approach: direct construction.
    
    # For the full 2x4*4x5 problem, let me try a Winograd-style approach
    # on the middle dimension.
    
    # Define the 4-point transform for the j dimension:
    # For each (i,k) pair, C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # 
    # We can use the 4-point FFT-like transform:
    # u0 = A[i,0] + A[i,1] + A[i,2] + A[i,3]
    # u1 = A[i,0] + A[i,1] - A[i,2] - A[i,3]
    # u2 = A[i,0] - A[i,1] + A[i,2] - A[i,3]
    # u3 = A[i,0] - A[i,1] - A[i,2] + A[i,3]
    # 
    # Similarly for B:
    # v0 = B[0,k] + B[1,k] + B[2,k] + B[3,k]
    # v1 = B[0,k] + B[1,k] - B[2,k] - B[3,k]
    # v2 = B[0,k] - B[1,k] + B[2,k] - B[3,k]
    # v3 = B[0,k] - B[1,k] - B[2,k] + B[3,k]
    # 
    # Then C[i,k] = (u0*v0 + u1*v1 + u2*v2 + u3*v3) / 4
    # 
    # This uses 4 multiplications per (i,k) pair, for 2*5 = 10 pairs, total 40 mults.
    # Same as schoolbook. Not helpful.
    
    # The key is to share computations across i and k.
    
    # Let me try the block-Strassen approach more carefully.
    # Split A into A_left (2x2) and A_right (2x2), B into B_top (2x5) and B_bottom (2x5).
    # 
    # C = A_left * B_top + A_right * B_bottom
    # 
    # Each term is 2x2 * 2x5.
    # 
    # For 2x2 * 2x5, use Strassen:
    # A = [[a,b],[c,d]], B = [[b1,b2,b3,b4,b5],[c1,c2,c3,c4,c5]]
    # 
    # Strassen:
    # p1 = (a+d)*(b1+c1)
    # p2 = (c+d)*(b1)
    # p3 = (a)*(b2-c2)
    # p4 = (d)*(c3-b3)  -- wait, this doesn't work for non-square B
    # 
    # Strassen is for square matrices. For 2x2 * 2x5, I need a different approach.
    
    # Let me just use the schoolbook for the 2x2*2x5 blocks:
    # C1 = A_left * B_top: 4 multiplications per output column, 5 columns = 20 mults
    # C2 = A_right * B_bottom: same, 20 mults
    # Total: 40 mults. Same as before.
    
    # To do better, I need to share the A_left and A_right computations.
    
    # Alternative: Use the n=2 structure to create a "dual" representation.
    # 
    # Let me try a different decomposition. Consider:
    # For each j, define the contribution to C:
    # C[:,k] += A[:,j] * B[j,k]
    # 
    # This is an outer product of a 2-vector and a scalar, for each j,k.
    # Total: 4*5 = 20 outer products, each of rank 1 in the "A*B" sense.
    # But we can combine them.
    
    # I think the best approach for now is to stick with the block-Strassen
    # decomposition that achieves rank 36, and try to optimize the merge step.
    
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

    # Build block-Strassen decomposition (rank 36)
    add_strassen_2x2x2((0, 0), (0, 0), (0, 0))
    add_strassen_2x2x2((0, 2), (2, 0), (0, 0))
    add_strassen_2x2x2((0, 0), (0, 2), (0, 2))
    add_schoolbook_2x2x1((0, 0), (0, 4), (0, 4))
    add_strassen_2x2x2((0, 2), (2, 2), (0, 2))
    add_schoolbook_2x2x1((0, 2), (2, 4), (0, 4))

    # Winograd-style merge: if two terms share identical U and identical W,
    # merge their V columns into one term.
    def merge_terms(terms):
        from collections import defaultdict
        groups = defaultdict(list)
        for idx, (u, v, w) in enumerate(terms):
            u_key = tuple(sorted(u.items()))
            w_key = tuple(sorted(w.items()))
            groups[(u_key, w_key)].append((idx, v))
        merged = []
        used = set()
        for (u_key, w_key), members in groups.items():
            if len(members) == 1:
                idx, v = members[0]
                if idx not in used:
                    u = dict(u_key)
                    w = dict(w_key)
                    merged.append((u, v, w))
                    used.add(idx)
            else:
                u = dict(u_key)
                w = dict(w_key)
                v_sum = {}
                for idx, v in members:
                    if idx in used:
                        continue
                    for (jk, coeff) in v.items():
                        v_sum[jk] = v_sum.get(jk, 0) + coeff
                    used.add(idx)
                v_sum = {k: val for k, val in v_sum.items() if val != 0}
                if v_sum:
                    merged.append((u, v_sum, w))
        return merged

    rank_terms = merge_terms(rank_terms)

    # Verify correctness
    def verify(terms):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for (u, v, w) in terms:
                        total += u.get((i, j), 0) * v.get((j, k), 0) * w.get((i, k), 0)
                    if total != 1:
                        return False
        return True

    if not verify(rank_terms):
        rank_terms = []
        add_strassen_2x2x2((0, 0), (0, 0), (0, 0))
        add_strassen_2x2x2((0, 2), (2, 0), (0, 0))
        add_strassen_2x2x2((0, 0), (0, 2), (0, 2))
        add_schoolbook_2x2x1((0, 0), (0, 4), (0, 4))
        add_strassen_2x2x2((0, 2), (2, 2), (0, 2))
        add_schoolbook_2x2x1((0, 2), (2, 4), (0, 4))

    rank = len(rank_terms)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][col] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][col] = coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
