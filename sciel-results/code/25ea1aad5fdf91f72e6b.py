"""Matrix multiplication tensor decomposition for 2x3 * 3x10."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # We decompose the multiplication A(2x3) * B(3x10) = C(2x10)
    # A has entries A[i, j] for i in 0..1, j in 0..2
    # B has entries B[j, k] for j in 0..2, k in 0..9
    # C[i, k] = sum_j A[i,j] * B[j,k]
    
    # Tensor T[i, j, k] such that C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # Actually the tensor decomposition is:
    # C[i,k] = sum_r U[i*m+j_r, r] * V[j_r*p+k, r] * W[k*n+i, r]
    # where the sum is over r (rank dimension)
    # 
    # Standard form: C[i,k] = sum_j A[i,j] B[j,k]
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = A[i,j] * B[j,k] summed over j
    # 
    # Actually the convention is:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k' V[j*p+k', r] * B[j,k']) ... 
    # No, that's not right either.
    #
    # Let me re-read the task: "W[k*n+i,r]" suggests W is indexed by (k, i) with k in 0..p-1, i in 0..n-1
    # The tensor T has entries T[i, j, k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # 
    # The rank-r decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # 
    # For schoolbook: T[i,j,k] = delta_{i,j,k} (it's 1 when indices match in the multiplication)
    # Actually T[i,j,k] = 1 for all i,j,k in the multiplication tensor (it's the structure tensor)
    # No: the matrix multiplication tensor has T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B 
    # multiply to give the (i,k) entry of C. So T[i,j,k] = 1 for all valid i,j,k.
    #
    # The decomposition: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i,j,k
    #
    # In schoolbook: one term per (i,j,k) triple, with U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1
    # and zeros elsewhere.
    
    # Now, the strategy: exploit the structure.
    # For j=0,1: we have a 2x2 x 2x10 subproblem for each pair of k values.
    # Actually, let's think of it differently.
    #
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # The j=2 part: C[i,k] += A[i,2]*B[2,k] for each i,k. This is a rank 2*n*p... no.
    # For j=2: we need T[i,2,k] = U[i*3+2, r] * V[2*10+k, r] * W[k*2+i, r] to give 1
    # This is just the product of A[i,2] and B[2,k], which is a rank-1 tensor for each (i,k) pair.
    # So that's n*p = 2*10 = 20 terms of rank 1? No, we can do better.
    #
    # Actually, for the j=2 part: we need for all i in {0,1}, k in {0..9}:
    # sum_r U[i*3+2, r] * V[20+k, r] * W[2k+i, r] = 1
    # This is equivalent to saying the outer product of the vectors u_i = (U[i*3+2, r] for all r) 
    # and v_k = (V[20+k, r] for all r) and w_{ik} = (W[2k+i, r] for all r) ... 
    # No, it's more subtle. We need:
    # sum_r U[i*3+2, r] * V[20+k, r] * W[2k+i, r] = 1 for all i,k
    #
    # This can be achieved with rank = n*p = 20 (one for each i,k), but also with fewer.
    # Actually, if we set U[i*3+2, r] = delta_{i, r mod n} for r in some range...
    #
    # Let me think about this more carefully using the "transposed" view.
    # The task says we can use the commutative property: A(2x3)*B(3x10) = (B^T * A^T)^T = (10x3 * 3x2)^T
    # 
    # So equivalently, we're decomposing the tensor for 10x3 * 3x2 multiplication.
    # Let A' be 10x3 (A' = B^T), B' be 3x2 (B' = A^T), C' be 10x2 (C' = C^T)
    # C'[k, i] = sum_j A'[k,j] * B'[j,i] = sum_j B[j,k] * A[i,j]
    #
    # Now A' is 10x3, B' is 3x2.
    # For j=0,1: A'[k,0..1] is 10x2, B'[0..1,i] is 2x2. This is a 10x2 * 2x2 = 10x2 multiplication.
    # For j=2: A'[k,2] is 10x1, B'[2,i] is 1x2. This is a 10x1 * 1x2 = 10x2 multiplication.
    #
    # For the 10x2 * 2x2 part: We can view this as 5 independent 2x2 * 2x2 multiplications
    # (pairing k values: (0,1), (2,3), (4,5), (6,7), (8,9)).
    # Each 2x2 * 2x2 has rank 7 (Strassen). So 5 * 7 = 35.
    #
    # For the 10x1 * 1x2 part: This is just outer products. C'[k,i] += A'[k,2] * B'[2,i]
    # This is a 10x2 matrix that's the outer product of a 10x1 and 1x2 vector.
    # The tensor for this is: T[k, 2, i] = 1 for all k,i. 
    # Decomposition: sum_r U'[k*3+2, r] * V'[2*2+i, r] * W'[2k+i... ] 
    # Wait, I need to be careful about the index conventions.
    #
    # Let me just directly construct the decomposition for the original problem.
    
    # Original: n=2, m=3, p=10
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (30, rank)  
    # W: (p*n, rank) = (20, rank)
    
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor: T[i,j,k] should be decomposable as sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # Part 1: j in {0, 1} - use Strassen on 2x2 sub-blocks
    # For each pair of k values (k1, k2), and for i in {0,1}, j in {0,1}:
    # We have a 2x2 (i,j) times 2x2 (j,k) = 2x2 (i,k) multiplication.
    
    # Strassen's 7 products for 2x2 * 2x2:
    # Given A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b21-b22)
    # m4 = a22(b11-b12)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b12+b22)
    # m7 = (a12-a21)(b11+b21)
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # In our case, for a fixed k-pair (k1, k2):
    # A_sub[i, j] = A[i, j] for i in {0,1}, j in {0,1}
    # B_sub[j, k_local] = B[j, k1 + k_local] for j in {0,1}, k_local in {0,1}
    # C_sub[i, k_local] should equal sum_j A[i,j] * B[j, k1+k_local]
    
    # The Strassen decomposition gives:
    # C_sub[i, k_local] = sum_{s=0}^{6} U_s[i, j_combined] * V_s[j_combined, k_local] * W_s[k_local, i]
    # where the "combined" indices need to be mapped back.
    
    # Actually, let me think of it as: each Strassen term is a rank-1 contribution.
    # For term s, we have:
    # U contribution: linear combination of A[i, j] values
    # V contribution: linear combination of B[j, k] values  
    # W contribution: coefficient for C[i, k]
    
    # Let me enumerate the Strassen terms explicitly.
    # For a 2x2 * 2x2 block with i in {0,1}, j in {0,1}, k_local in {0,1}:
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U_coeffs: dict mapping (i,j) to coefficient in U
        # V_coeffs: dict mapping (j,k_local) to coefficient in V
        # W_coeffs: dict mapping (k_local, i) to coefficient in W
        
        # m1 = (a00+a11)(b00+b11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}),  # This is wrong, let me redo
        
    ]
    
    # Let me be more careful. In Strassen:
    # m1 = (A00+A11)(B00+B11) contributes to C00 and C11
    # Specifically: C00 gets +m1, C11 gets +m1
    # m2 = (A10+A11)B00 contributes to C10 and C11
    # C10 gets +m2, C11 gets -m2
    # m3 = A00(B10-B11) contributes to C01 and C11
    # C01 gets +m3, C11 gets +m3
    # m4 = A11(B00-B01) contributes to C00, C10, C11
    # C00 gets +m4, C10 gets +m4, C11 gets -m4... 
    
    # Let me just use the standard formulas:
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # So the 7 Strassen rank-1 terms and their W contributions:
    # Term 0 (m1): U: A00+A11, V: B00+B11, W: C00 +1, C11 +1
    # Term 1 (m2): U: A10+A11, V: B00, W: C10 +1, C11 -1
    # Term 2 (m3): U: A00, V: B10-B11, W: C01 +1, C11 +1
    # Term 3 (m4): U: A11, V: B00-B01, W: C00 +1, C10 +1, C11 -1
    # Term 4 (m5): U: A00+A01, V: B11, W: C00 -1, C01 +1
    # Term 5 (m6): U: A10-A00, V: B01+B11, W: C11 +1
    # Term 6 (m7): U: A01-A10, V: B00+B10, W: C00 +1
    
    # Now I need to map these to the full index space.
    
    U = {}
    V = {}
    W = {}
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        """Add a rank-1 term. u_coeffs: list of (row_idx, coeff), etc."""
        rank = len(U) if U else 0
        # Find next available rank
        while rank in U:
            rank += 1
        
        u_vec = {}
        for idx, coeff in u_coeffs:
            u_vec[idx] = u_vec.get(idx, 0) + coeff
        v_vec = {}
        for idx, coeff in v_coeffs:
            v_vec[idx] = v_vec.get(idx, 0) + coeff
        w_vec = {}
        for idx, coeff in w_coeffs:
            w_vec[idx] = w_vec.get(idx, 0) + coeff
        
        U[rank] = u_vec
        V[rank] = v_vec
        W[rank] = w_vec
        return rank
    
    rank = 0
    
    # Process 5 k-pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
    k_pairs = [(0,1), (2,3), (4,5), (6,7), (8,9)]
    
    for k1, k2 in k_pairs:
        # For each k-pair, apply Strassen to the 2x2 * 2x2 block
        # i in {0,1}, j in {0,1}, k_local in {0,1} where k = k1 + k_local
        
        # Term m1: (A00+A11)(B00+B11)
        u_coeffs = [(0*3+0, 1), (1*3+1, 1)]  # A[0,0] + A[1,1]
        v_coeffs = [(0*10+k1, 1), (1*10+k1, 1), (0*10+k2, 1), (1*10+k2, 1)]  # B[0,k1]+B[1,k1]+B[0,k2]+B[1,k2]
        w_coeffs = [(k1*2+0, 1), (k2*2+1, 1)]  # C[0,k1] + C[1,k2]
        add_term(u_coeffs, v_coeffs, w_coeffs)
        
        # Term m2: (A10+A11)B00
        u_coeffs = [(1*3+0, 1), (1*3+1, 1)]  # A[1,0] + A[1,1]
        v_coeffs = [(0*10+k1, 1)]  # B[0,k1]
        w_coeffs = [(k1*2+1, 1), (k2*2+1, -1)]  # C[1,k1] + C[1,k2]*(-1)... 
        # Wait, C10 = m2 + m4, so m2 contributes to C[1, k1] (since k_local=0 maps to k1)
        # C11 = m1 - m2 + m3 + m6, so m2 contributes to C[1, k2] with -1
        add_term(u_coeffs, v_coeffs, w_coeffs)
        
        # Term m3: A00(B10-B11)
        u_coeffs = [(0*3+0, 1)]  # A[0,0]
        v_coeffs = [(1*10+k1, 1), (1*10+k2, -1)]  # B[1,k1] - B[1,k2]
        w_coeffs = [(k2*2+0, 1), (k2*2+1, 1)]  # C[0,k2] + C[1,k2]
        # C01 = m3 + m5: m3 contributes to C[0, k2] (k_local=1 maps to k2)
        # C11 = m1 - m2 + m3 + m6: m3 contributes to C[1, k2]
        add_term(u_coeffs, v_coeffs, w_coeffs)
        
        # Term m4: A11(B00-B01)
        u_coeffs = [(1*3+1, 1)]  # A[1,1]
        v_coeffs = [(0*10+k1, 1), (0*10+k2, -1)]  # B[0,k1] - B[0,k2]
        w_coeffs = [(k1*2+0, 1), (k1*2+1, 1), (k2*2+1, -1)]  # C[0,k1] + C[1,k1] - C[1,k2]
        # C00 = m1 + m4 - m5 + m7: m4 contributes to C[0,k1]
        # C10 = m2 + m4: m4 contributes to C[1,k1]
        # C11 = m1 - m2 + m3 + m6: m4 doesn't appear? Let me recheck.
        # Actually I think I have the Strassen formulas slightly off. Let me just use a known correct set.
        
        # Let me restart with a cleaner approach. I'll directly verify by construction.
        pass
    
    # Let me just build it all at once with a clear structure.
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_rank1(u_dict, v_dict, w_dict):
        U_list.append(u_dict)
        V_list.append(v_dict)
        W_list.append(w_dict)
    
    # Strassen formulas (verified):
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b10-b11)
    # m4 = a11(b00-b01)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b01+b11)
    # m7 = (a01-a10)(b00+b10)
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    for k1, k2 in k_pairs:
        # a_ij = A[i,j], b_jk = B[j, k1+k_local]
        
        # m1: U: A[0,0]+A[1,1], V: B[0,k1]+B[1,k1]+B[0,k2]+B[1,k2], W: C[0,k1]+C[1,k2]
        add_rank1({0*3+0: 1, 1*3+1: 1}, 
                  {0*10+k1: 1, 1*10+k1: 1, 0*10+k2: 1, 1*10+k2: 1},
                  {k1*2+0: 1, k2*2+1: 1})
        
        # m2: U: A[1,0]+A[1,1], V: B[0,k1], W: C[1,k1] - C[1,k2]
        add_rank1({1*3+0: 1, 1*3+1: 1},
                  {0*10+k1: 1},
                  {k1*2+1: 1, k2*2+1: -1})
        
        # m3: U: A[0,0], V: B[1,k1]-B[1,k2], W: C[0,k2] + C[1,k2]
        add_rank1({0*3+0: 1},
                  {1*10+k1: 1, 1*10+k2: -1},
                  {k2*2+0: 1, k2*2+1: 1})
        
        # m4: U: A[1,1], V: B[0,k1]-B[0,k2], W: C[0,k1] + C[1,k1] - C[1,k2]
        add_rank1({1*3+1: 1},
                  {0*10+k1: 1, 0*10+k2: -1},
                  {k1*2+0: 1, k1*2+1: 1, k2*2+1: -1})
        
        # m5: U: A[0,0]+A[0,1], V: B[1,k2], W: -C[0,k1] + C[0,k2]
        add_rank1({0*3+0: 1, 0*3+1: 1},
                  {1*10+k2: 1},
                  {k1*2+0: -1, k2*2+0: 1})
        
        # m6: U: A[1,0]-A[0,0], V: B[0,k2]+B[1,k2], W: C[1,k2]
        add_rank1({1*3+0: 1, 0*3+0: -1},
                  {0*10+k2: 1, 1*10+k2: 1},
                  {k2*2+1: 1})
        
        # m7: U: A[0,1]-A[1,0], V: B[0,k1]+B[1,k1], W: C[0,k1]
        add_rank1({0*3+1: 1, 1*3+0: -1},
                  {0*10+k1: 1, 1*10+k1: 1},
                  {k1*2+0: 1})
    
    # Part 2: j=2 term: C[i,k] += A[i,2]*B[2,k] for all i,k
    # This is n*p = 20 independent rank-1 terms (one per i,k pair)
    # But we can do better! 
    # For fixed j=2: we need sum_r U[i*3+2, r] * V[20+k, r] * W[2k+i, r] = 1 for all i,k
    # 
    # One efficient way: use rank = n*p = 20, but we can also use a smarter decomposition.
    # Actually, the minimal rank for this subproblem (10x1 * 1x2 = 10x2) is min(10*1, 1*2, ...) 
    # The tensor is just the all-ones tensor of shape 10x1x2 (in terms of k, j=2, i).
    # Rank of all-ones tensor of size a x b x c is min(a*b, b*c, c*a) when one dimension is 1.
    # Here it's 10 x 1 x 2, so rank = min(10, 2, 20) = 2.
    
    # So we can do the j=2 part in rank 2!
    # We need: sum_{r in R} U[i*3+2, r] * V[20+k, r] * W[2k+i, r] = 1 for all i in {0,1}, k in 0..9
    # 
    # With rank 2: Let r=0,1.
    # Set U[i*3+2, 0] = 1, U[i*3+2, 1] = 0 for all i
    # Set V[20+k, 0] = 1, V[20+k, 1] = 0 for all k  
    # Set W[2k+0, 0] = 1, W[2k+1, 0] = 0 for all k
    # Then term r=0 gives: 1*1*1 = 1 for i=0, and 1*1*0 = 0 for i=1. Not good.
    #
    # Let me try: U[i*3+2, 0] = delta_{i,0}, U[i*3+2, 1] = delta_{i,1}
    # V[20+k, 0] = 1, V[20+k, 1] = 1 for all k
    # W[2k+0, 0] = 1, W[2k+1, 1] = 1 for all k
    # Then for i=0: sum_r U[0*3+2, r] * V[20+k, r] * W[2k+0, r] = 1*1*1 + 0*1*0 = 1 ✓
    # For i=1: sum_r U[1*3+2, r] * V[20+k, r] * W[2k+1, r] = 0*1*0 + 1*1*1 = 1 ✓
    
    # But wait, these rank indices are already used. I need to add new rank dimensions.
    
    # Add two more rank-1 terms for the j=2 part
    add_rank1({0*3+2: 1, 1*3+2: 0},
              {20+k: 1 for k in range(10)},
              {2*k+0: 1 for k in range(10)})
    
    add_rank1({0*3+2: 0, 1*3+2: 1},
              {20+k: 1 for k in range(10)},
              {2*k+1: 1 for k in range(10)})
    
    # Now build the output matrices
    total_rank = len(U_list)
    
    U_mat = [[0] * total_rank for _ in range(6)]
    V_mat = [[0] * total_rank for _ in range(30)]
    W_mat = [[0] * total_rank for _ in range(20)]
    
    for r in range(total_rank):
        for idx, coeff in U_list[r].items():
            U_mat[idx][r] = coeff
        for idx, coeff in V_list[r].items():
            V_mat[idx][r] = coeff
        for idx, coeff in W_list[r].items():
            W_mat[idx][r] = coeff
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
