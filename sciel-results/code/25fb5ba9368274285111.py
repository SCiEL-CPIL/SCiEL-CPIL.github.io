"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # We'll build up U, V, W as lists of columns (each column is a vector)
    # U has shape (n*m, rank) = (20, rank)
    # V has shape (m*p, rank) = (30, rank)
    # W has shape (p*n, rank) = (24, rank)
    
    U_cols = []  # list of vectors of length n*m
    V_cols = []  # list of vectors of length m*p
    W_cols = []  # list of vectors of length p*n
    
    def add_factor(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    def zero_vec(length):
        return [0] * length
    
    def set_entry(vec, idx, val):
        vec[idx] = vec[idx] + val
    
    # Schoolbook helper for (n1 x m1) * (m1 x p1)
    # Adds (n1*m1*p1) factors
    # i ranges over [0, n1), j over [0, m1), k over [0, p1)
    # Global indices: i_off + i, j_off + j, k_off + k
    def add_schoolbook(n1, m1, p1, i_off, j_off, k_off):
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    u = zero_vec(n * m)
                    v = zero_vec(m * p)
                    w = zero_vec(p * n)
                    set_entry(u, (i_off + i) * m + (j_off + j), 1)
                    set_entry(v, (j_off + j) * p + (k_off + k), 1)
                    set_entry(w, (k_off + k) * n + (i_off + i), 1)
                    add_factor(u, v, w)
    
    # Strassen helper for (n1 x m1) * (m1 x p1) where n1=m1=p1=2
    # Uses rank-7 decomposition
    # i ranges over [0, n1), j over [0, m1), k over [0, p1)
    # Global indices: i_off + i, j_off + j, k_off + k
    def add_strassen_2x2(i_off, j_off, k_off):
        # Strassen's 7 products for 2x2 * 2x2
        # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b10)
        # M7 = (a01-a11)(b01+b11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For each of the 7 terms, we need to express as sum over j of U*V*W
        # Term r: C[i,k] += coeff * A[i, j_a] * B[j_b, k]
        # But Strassen combines multiple A entries and multiple B entries in each product.
        # So each Strassen term is actually a sum of products, which we need to expand.
        
        # M1 = (a00+a11)(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
        # These contribute to C00, C11 with coefficients
        
        # Let me define each Strassen product as a set of (i, j_a, j_b, k, coeff_C) contributions
        # Actually, each Strassen term M_r = sum_{i,j_a} U_ri * A[i,j_a] * sum_{j_b,k} V_rk * B[j_b,k]
        # where the product M_r contributes to C with a pattern.
        
        # Better approach: express each of the 7 Strassen products as a tensor decomposition term.
        # Each product is of the form: M_r = (sum_{i,ja} alpha_{r,ij_a} * A[i,ja]) * (sum_{jb,k} beta_{r,jb,k} * B[jb,k])
        # And M_r contributes to C as: sum_{i,k} gamma_{r,ik} * M_r * C[i,k]
        
        # Actually the decomposition is: C[i,k] = sum_r (sum_{ja} U[i,ja,r] * A[i,ja]) * (sum_{jb} V[jb,k,r] * B[jb,k]) * W[k,i,r]
        # No, that's not right. The standard form is:
        # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # So for a Strassen term, we need to assign U, V, W values such that:
        # sum_j U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r] = M_r's contribution to C[i,k]
        
        # For M1: M1 = (a00+a11)(b00+b11)
        # M1 contributes: +1*M1 to C00, +1*M1 to C11
        # We need: sum_j U[i*m+j, r1] * V[j*p+k, r1] * W[k*n+i, r1] = [M1's contribution pattern]
        # M1's contribution to C[i,k]:
        # C00 += M1, C11 += M1, others 0
        # M1 = (a00+a11)(b00+b11) = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
        # = A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]
        
        # For the decomposition to work, we need:
        # sum_j U[0*m+j, r] * V[j*p+0, r] * W[0*n+0, r] = 1  (for C00)
        # sum_j U[1*m+j, r] * V[j*p+1, r] * W[1*n+1, r] = 1  (for C11)
        # and the product U[i*m+j, r] * V[j*p+k, r] must match the coefficient pattern.
        
        # This is getting complex. Let me use a different approach:
        # Expand each Strassen product into its constituent rank-1 terms.
        
        # M1 = (a00+a11)(b00+b11)
        # This is a product of a linear form in A and a linear form in B.
        # As a bilinear form, it's rank 4 (it's a sum of 4 rank-1 terms).
        # But in the tensor decomposition, each "term" in the sum over r can handle
        # multiple (i,j,k) entries.
        
        # Let me think differently. The Strassen algorithm gives us 7 bilinear forms.
        # Each bilinear form M_r can be written as:
        # M_r = sum_{i,j} U_{r,ij} * A[i,j] * sum_{j',k} V_{r,j'k} * B[j',k]
        # where U_{r,ij} and V_{r,j'k} are the coefficients.
        # And M_r contributes to C as:
        # C[i,k] += sum_r W_{r,ki} * M_r  (note: W[k*n+i, r] = W_{r,ki})
        
        # So for each Strassen product r, we need:
        # U_{r,ij} for i in {0,1}, j in {0,1}  (relative to block)
        # V_{r,j'k} for j' in {0,1}, k in {0,1}
        # W_{r,ki} for k in {0,1}, i in {0,1}
        
        # And the condition is:
        # sum_{j,j'} U_{r,ij} * V_{r,j'k} * (A[i,j]*B[j',k] when j=j' summed) ... 
        # No, that's not right either.
        
        # Let me reconsider. The standard matrix multiplication tensor decomposition is:
        # T_{ikj} = delta_{ij} delta_{jk}  (where T[i,k,j] means C[i,k] gets A[i,j]*B[j,k])
        # Decomposition: T_{ikj} = sum_r U_{ir} * V_{kr} * W_{jr}  (with appropriate index mapping)
        
        # In our case: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # The "tensor" being decomposed has indices (i, j, k) with value 1 if they form a valid term.
        # Actually the tensor T[i,j,k] = 1 for all i,j,k (it's the identity tensor for matmul).
        # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Wait, that's not quite right. Let me re-examine.
        
        # The matrix multiplication map is: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
        # This is a trilinear map. The tensor for this map has entries:
        # T[i, j, k] = 1 (for the term A[i,j]*B[j,k] contributing to C[i,k])
        # So T is a 3-tensor with T[i,j,k] = 1 for all valid i,j,k.
        
        # A rank-r decomposition is: T[i,j,k] = sum_{r=1}^r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Wait, that doesn't make sense dimensionally. Let me look at the seed code again.
        
        # Seed: For each (i,j,k): U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
        # So T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For schoolbook: T[i,j,k] = 1 (only the term where r corresponds to (i,j,k) is 1)
        # So yes, T[i,j,k] = 1 for all i,j,k.
        
        # Now for Strassen: we want to replace the 8 rank-1 terms (schoolbook for 2x2x2)
        # with 7 terms. Each term r has:
        # U[i*m+j, r] for i,j in {0,1}
        # V[j*p+k, r] for j,k in {0,1}  
        # W[k*n+i, r] for k,i in {0,1}
        # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k in {0,1}
        
        # Strassen's 7 products:
        # Let's index the 2x2 matrices with local indices 0,1
        # A = [[a0, a1], [a2, a3]] where a0=A[0,0], a1=A[0,1], a2=A[1,0], a3=A[1,1]
        # B = [[b0, b1], [b2, b3]] where b0=B[0,0], b1=B[0,1], b2=B[1,0], b3=B[1,1]
        # C = [[c0, c1], [c2, c3]] where c0=C[0,0], c1=C[0,1], c2=C[1,0], c3=C[1,1]
        
        # M1 = (a0+a3)(b0+b3)
        # M2 = (a2+a3)b0
        # M3 = a0(b2-b3)
        # M4 = a3(b1-b0)
        # M5 = (a0+a1)b3
        # M6 = (a2-a0)(b0+b2)
        # M7 = (a1-a3)(b1+b3)
        
        # c0 = M1 + M4 - M5 + M7
        # c1 = M3 + M5
        # c2 = M2 + M4
        # c3 = M1 - M2 + M3 + M6
        
        # For each M_r, it's a product of a linear form in A and a linear form in B.
        # M_r = (sum_{ij} alpha_{r,ij} * A[i,j]) * (sum_{jk} beta_{r,jk} * B[j,k])
        # And M_r contributes to C[i,k] with coefficient gamma_{r,ik}.
        
        # In the decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        # This means: for each (i,j,k), the sum over r of the product equals 1.
        
        # For a given Strassen term r:
        # U[i*m+j, r] = alpha_{r,ij} (coefficient of A[i,j] in the first factor)
        # V[j*p+k, r] = beta_{r,jk} (coefficient of B[j,k] in the second factor)
        # But then U*V*W for fixed (i,j,k) is alpha_{r,ij} * beta_{r,jk} * W[k*n+i, r]
        # And we need sum_r alpha_{r,ij} * beta_{r,jk} * W_{r,ki} = 1 for all i,j,k
        
        # Hmm, but alpha and beta depend on which A and B entries, and W depends on which C entry.
        # The issue is that a single Strassen term M_r involves multiple A entries and multiple B entries.
        # So alpha_{r,ij} is nonzero for multiple (i,j), and beta_{r,jk} is nonzero for multiple (j,k).
        
        # For M1 = (a0+a3)(b0+b3):
        # alpha_{1,00} = 1, alpha_{1,11} = 1, others 0
        # beta_{1,00} = 1, beta_{1,11} = 1, others 0
        # M1 contributes to c0 with +1, c3 with +1
        # So W_{1,00} = 1 (for c0), W_{1,33} = 1 (for c3), but W is indexed by (k,i) -> k*n+i
        # W[k*n+i, r] where k is col of C, i is row of C.
        # c0 is C[0,0]: k=0, i=0, so W[0*n+0, 1] = W[0,1] = 1
        # c3 is C[1,1]: k=1, i=1, so W[1*n+1, 1] = W[n+1, 1] = 1
        # But wait, M1 = (a0+a3)(b0+b3) = a0*b0 + a0*b3 + a3*b0 + a3*b3
        # The product U*V for (i=0,j=0): alpha_{1,00} * beta_{1,00} = 1*1 = 1, times W[0*n+0,1] = 1 -> contributes 1 to T[0,0,0]
        # For (i=0,j=1): alpha_{1,01} = 0, so contributes 0
        # For (i=1,j=0): alpha_{1,10} = 0, so contributes 0
        # For (i=1,j=1): alpha_{1,11} = 1, beta_{1,11} = 1, W[1*n+1,1] = 1 -> contributes 1 to T[1,1,1]
        # But we also need T[0,0,1] = 1 (a0*b3) and T[1,1,0] = 1 (a3*b0)
        # For (i=0,j=0,k=1): alpha_{1,00}=1, beta_{1,01}=0 (since beta only has 00 and 11), so contributes 0. Problem!
        
        # I see the issue. The Strassen product M1 = (a0+a3)(b0+b3) is a single scalar, but it's formed from
        # multiple A-B pairs. In the tensor decomposition, each (i,j,k) triple must sum to 1.
        # The product M1 as a whole contributes to c0 and c3, but the intermediate (i,j,k) terms don't all work out.
        
        # I think the correct way is: each Strassen term is NOT a single rank-1 term in the tensor decomposition.
        # Instead, each Strassen product M_r is itself a bilinear form that can be decomposed.
        # Actually, I think I'm overcomplicating this.
        
        # Let me use a simpler approach: just use schoolbook for all sub-blocks, but that gives rank 120.
        # To reduce rank, I need to actually use the Strassen identities correctly.
        
        # Alternative: use the known optimal or near-optimal decompositions.
        # For 4x5 x 5x6, what's the known lower bound?
        # The rank of matrix multiplication tensor for n x m * m x p is at most n*m*p (schoolbook).
        # For small sizes, there are known improvements.
        
        # Let me try a different decomposition strategy:
        # Split the 5 into 2+3, and 4 into 2+2, 6 into 3+3.
        # A is 4x5: split into rows [0,1] and [2,3], cols [0,1,2] and [3,4]
        # B is 5x6: split into rows [0,1,2] and [3,4], cols [0,1,2] and [3,4,5]
        
        # A11: 2x3, A12: 2x2, A21: 2x3, A22: 2x2
        # B11: 3x3, B12: 3x3, B21: 2x3, B22: 2x3
        
        # C11 = A11*B11 + A12*B21: (2x3)(3x3) + (2x2)(2x3) -> 2x3
        # C12 = A11*B12 + A12*B22: (2x3)(3x3) + (2x2)(2x3) -> 2x3
        # C21 = A21*B11 + A22*B21: (2x3)(3x3) + (2x2)(2x3) -> 2x3
        # C22 = A21*B12 + A22*B22: (2x3)(3x3) + (2x2)(2x3) -> 2x3
        
        # Schoolbook ranks:
        # (2x3)(3x3): 2*3*3 = 18
        # (2x2)(2x3): 2*2*3 = 12
        # Each C block needs 18+12 = 30, four blocks: 120. Same as before.
        
        # I need to actually reduce the rank. Let me look up or derive a better decomposition.
        
        # For now, let me implement a decomposition that at least is correct, even if rank=120.
        # But the task wants rank-minimal, so I should try harder.
        
        # Let me try: use Strassen for the 2x2x2 sub-problems where possible.
        
        # A12 is 2x2, B21 is 2x3. Not square, so Strassen doesn't directly apply.
        
        # What if I split differently? 
        # A: 4x5 -> rows 2+2, cols 2+3
        # B: 5x6 -> rows 2+3, cols 2+4
        # A11: 2x2, A12: 2x3, A21: 2x2, A22: 2x3
        # B11: 2x2, B12: 2x4, B21: 3x2, B22: 3x4
        # C11 = A11*B11 + A12*B21: (2x2)(2x2) + (2x3)(3x2) -> 2x2
        # C12 = A11*B12 + A12*B22: (2x2)(2x4) + (2x3)(3x4) -> 2x4
        # C21 = A21*B11 + A22*B21: (2x2)(2x2) + (2x3)(3x2) -> 2x2
        # C22 = A21*B12 + A22*B22: (2x2)(2x4) + (2x3)(3x4) -> 2x4
        
        # Schoolbook:
        # (2x2)(2x2): 8, (2x3)(3x2): 12 -> C11: 20
        # (2x2)(2x4): 16, (2x3)(3x4): 24 -> C12: 40
        # (2x2)(2x2): 8, (2x3)(3x2): 12 -> C21: 20
        # (2x2)(2x4): 16, (2x3)(3x4): 24 -> C22: 40
        # Total: 120. Still the same.
        
        # The schoolbook rank is always n*m*p regardless of how you split.
        # To reduce rank, I need to use algebraic identities that combine terms.
        
        # Let me just implement the schoolbook decomposition correctly and return rank 120.
        # This is the same as the seed, so it won't improve. But I need to at least produce a valid output.
        
        # Actually, let me try to implement a correct Strassen-based decomposition.
        # I'll handle the 2x2x2 blocks with Strassen and the rest with schoolbook,
        # being careful about how the terms combine.
        
        # For now, return schoolbook to ensure correctness.
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

