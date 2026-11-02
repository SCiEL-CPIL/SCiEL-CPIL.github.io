"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Block decomposition: 5 = 2+2+1
    # Use Strassen for 2x2 blocks, schoolbook for mixed sizes.
    # Block structure:
    #   A = [[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]]
    #   B = [[B11, B12, B13], [B21, B22, B23], [B31, B32, B33]]
    #   C = A*B = [[C11, C12, C13], [C21, C22, C23], [C31, C32, C33]]
    #
    # Block sizes:
    #   A11, A12, A21, A22 are 2x2
    #   A13, A23, A31, A32 are 2x1 or 1x2
    #   A33 is 1x1
    # Similarly for B and C.
    #
    # Strassen 2x2 decomposition (7 multiplications):
    #   M1 = (A11+A22)*(B11+B22)
    #   M2 = (A21+A22)*B11
    #   M3 = A11*(B12-B22)
    #   M4 = A22*(B21-B11)
    #   M5 = (A11+A12)*B22
    #   M6 = (A21-A11)*(B11+B12)
    #   M7 = (A12-A22)*(B21+B22)
    #   C11 = M1 + M4 - M5 + M7
    #   C12 = M3 + M5
    #   C21 = M2 + M4
    #   C22 = M1 - M2 + M3 + M6
    #
    # For mixed-size blocks, we pad to 2x2 where useful, or use direct schoolbook.
    # Strategy: group all 2x2 block multiplications that appear in C11, C12, C21, C22
    # and use Strassen. For other blocks (involving 1-wide blocks), use schoolbook.
    
    import time
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    # Block sizes: 2, 2, 1
    bs = [2, 2, 1]
    nb = 3  # number of blocks
    
    # We'll build U, V, W matrices.
    # U has shape (n*m, rank) = (25, rank)
    # V has shape (m*p, rank) = (25, rank)
    # W has shape (p*n, rank) = (25, rank)
    
    # Strategy: enumerate all "multiplication instructions" needed.
    # Each instruction is a pair of linear combinations of A-blocks and B-blocks
    # that produces a linear combination of C-blocks.
    
    # Let's think of this differently. We'll enumerate basis elements for U, V, W.
    # Each column of U is a linear combination of A entries.
    # Each column of V is a linear combination of B entries.
    # Each column of W is a linear combination of C entries (with the W convention W[k*n+i, r]).
    
    # The tensor for matrix multiplication C = A*B is:
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal delta[i,k] * ... wait, no.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has T[i,j,k] = 1 if we think of it as the identity in the (i,k) direction?
    # Actually, the matrix multiplication tensor M_{n,m,p} has:
    # M_{n,m,p}[i,j,k] = 1 for all i,j,k? No.
    # The tensor for C = AB is: C_{ik} = sum_j A_{ij} B_{jk}
    # As a 3-tensor T_{ijk}, T_{ijk} = 1 for all i,j,k? That's not right either.
    # Actually, T is defined by: C_{ik} = sum_j T_{ijk} A_{ij} B_{jk}
    # So T_{ijk} = 1 for all i,j,k. Wait, that gives C_{ik} = sum_j A_{ij} B_{jk} which is correct.
    # So T_{ijk} = 1 for all i,j,k in range(n,m,p).
    # Hmm, but that's a rank-1 tensor only if n=m=p=1.
    # The matrix multiplication tensor has entries T_{ijk} = delta_{ik}? No.
    # Let me reconsider. C_{ik} = sum_j A_{ij} B_{jk}.
    # So the bilinear form is: C_{ik} = sum_{j} A_{ij} B_{jk}.
    # As a tensor in the space of trilinear forms: T(A,B,C) = sum_{i,j,k} T_{ijk} A_{ij} B_{jk} C_{ki}... no.
    # The standard convention: the matrix multiplication tensor M has M_{ijk} = 1 for all i,j,k,
    # and the condition is that sum_{i,j,k} M_{ijk} U_i V_j W_k = 0 for all U,V in the nullspace.
    # Actually, for the factorization to be exact:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should reconstruct the multiplication.
    # The multiplication tensor is: for each (i,j,k), the contribution to C[i,k] from A[i,j]*B[j,k].
    # So we need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Wait, that's not quite right either. Let me think again.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # If we write A[i,j] = sum_r U[i*m+j, r] * x_r and B[j,k] = sum_r V[j*p+k, r] * y_r,
    # then A[i,j]B[j,k] = sum_{r,s} U[i*m+j,r] V[j*p+k,s] x_r y_s.
    # We want C[i,k] = sum_j A[i,j] B[j,k] = sum_r W[k*n+i, r] * (something).
    # Actually, the standard tensor decomposition for matrix multiplication:
    # The tensor T = sum_{i,j,k} e_i (in C-space) ⊗ e_{ij} (in A-space) ⊗ e_{jk} (in B-space)
    # where the C-space index is (k,n+i) in the W convention? 
    # Let me use the convention from the task: W[k*n+i, r].
    # So the output C has entries C[i,k] for i in 0..n-1, k in 0..p-1.
    # W is indexed by (k*n+i), so W row index = k*n + i.
    # The tensor decomposition: for each r, we have U[:,r], V[:,r], W[:,r].
    # The product is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal the "multiplication kernel" which is:
    # For the term A[i,j] * B[j,k] to contribute to C[i,k], we need the sum over r to give 1
    # when the indices match appropriately. Actually, the tensor M_{n,m,p} is defined by:
    # M_{n,m,p} = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_{i,k} ⊗ e_{i,j} ⊗ e_{j,k}
    # where e_{i,k} is the basis vector in C-space (indexed by k*n+i), e_{i,j} in A-space (i*m+j),
    # e_{j,k} in B-space (j*p+k).
    # So M_{n,m,p}[k*n+i, i*m+j, j*p+k] = 1 for all i,j,k, and 0 otherwise.
    # The rank is the minimum number of terms in a decomposition of this tensor.
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    # And for other index combinations, it should be 0, but since the tensor only has non-zero entries
    # at (k*n+i, i*m+j, j*p+k), we just need the sum to equal 1 at those positions.
    
    # Approach: build a list of (U_vec, V_vec, W_vec) triples.
    # Each triple contributes to specific (i,j,k) positions.
    
    # Let's use a block-based approach.
    # Block indices: 0, 1, 2 with sizes 2, 2, 1.
    # Block b covers rows/cols [offset_b, offset_b + size_b).
    # offsets: [0, 2, 4]
    # sizes: [2, 2, 1]
    
    offsets = [0, 2, 4]
    sizes = [2, 2, 1]
    
    # We'll collect (u, v, w) where u is a dict mapping A-index (i,j) to coefficient,
    # v maps B-index (j,k) to coefficient, w maps C-index (i,k) to coefficient.
    # Then we convert to vectors.
    
    terms = []  # list of (u_dict, v_dict, w_dict)
    
    # Helper: add a term
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
    
    # Strassen for 2x2 block multiplication.
    # Blocks: A_ab (rows a, cols b), B_bc (rows b, cols c), C_ac (rows a, cols c)
    # where a, b, c are block indices (0 or 1 for 2x2 blocks).
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For Strassen on blocks a, b, c (each 2x2):
    # A blocks: A_ab, A_a(b+1), A_(a+1)b, A_(a+1)(b+1)
    # B blocks: B_bc, B_b(c+1), B_(b+1)c, B_(b+1)(c+1)
    # C blocks: C_ac, A_a(c+1), C_(a+1)c, C_(a+1)(c+1)
    
    def strassen_2x2(a, b, c):
        """Add 7 Strassen terms for 2x2 block multiplication."""
        # M1 = (A_a0b0 + A_a1b1) * (B_b0c0 + B_b1c1)
        # In terms of individual entries:
        # A_a0b0 = A[off_a+0, off_b+0], etc.
        off_a, off_b, off_c = offsets[a], offsets[b], offsets[c]
        
        # M1
        u = {}
        v = {}
        w = {}
        # A part: A[off_a, off_b] + A[off_a+1, off_b+1]
        u[(off_a, off_b)] = 1
        u[(off_a+1, off_b+1)] = 1
        # B part: B[off_b, off_c] + B[off_b+1, off_c+1]
        v[(off_b, off_c)] = 1
        v[(off_b+1, off_c+1)] = 1
        # C part: C[off_a, off_c] + C[off_a+1, off_c+1] - C[off_a, off_c+1] + C[off_a+1, off_c]
        # C00 = M1 + M4 - M5 + M7, so M1 contributes +1 to C00
        w[(off_a, off_c)] = 1
        # We'll accumulate contributions separately. Let's just add M1 as a term with
        # the C coefficients as they appear in C00, C01, C10, C11.
        # Actually, each M contributes to multiple C blocks. Let me restructure.
        # Each M is a scalar multiplication. The C blocks are linear combinations of Ms.
        # So for each M, we add a term where:
        #   u is the linear combo of A entries
        #   v is the linear combo of B entries
        #   w is the linear combo of C entries (with signs)
        
        # M1: A=(A00+A11), B=(B00+B11)
        # C contributions: C00 += 1*M1, C11 += 1*M1
        add_term(
            {(off_a, off_b): 1, (off_a+1, off_b+1): 1},
            {(off_b, off_c): 1, (off_b+1, off_c+1): 1},
            {(off_a, off_c): 1, (off_a+1, off_c+1): 1}
        )
        
        # M2: A=(A10+A11), B=B00
        # C contributions: C10 += 1*M2, C11 += -1*M2
        add_term(
            {(off_a+1, off_b): 1, (off_a+1, off_b+1): 1},
            {(off_b, off_c): 1},
            {(off_a+1, off_c): 1, (off_a+1, off_c+1): -1}
        )
        
        # M3: A=A00, B=(B01-B11)
        # C contributions: C01 += 1*M3, C11 += 1*M3
        add_term(
            {(off_a, off_b): 1},
            {(off_b, off_c+1): 1, (off_b+1, off_c+1): -1},
            {(off_a, off_c+1): 1, (off_a+1, off_c+1): 1}
        )
        
        # M4: A=A11, B=(B10-B00)
        # C contributions: C00 += 1*M4, C10 += 1*M4
        add_term(
            {(off_a+1, off_b+1): 1},
            {(off_b+1, off_c): 1, (off_b, off_c): -1},
            {(off_a, off_c): 1, (off_a+1, off_c): 1}
        )
        
        # M5: A=(A00+A01), B=B11
        # C contributions: C00 += -1*M5, C01 += 1*M5
        add_term(
            {(off_a, off_b): 1, (off_a, off_b+1): 1},
            {(off_b+1, off_c+1): 1},
            {(off_a, off_c): -1, (off_a, off_c+1): 1}
        )
        
        # M6: A=(A10-A00), B=(B00+B01)
        # C contributions: C11 += 1*M6
        add_term(
            {(off_a+1, off_b): 1, (off_a, off_b): -1},
            {(off_b, off_c): 1, (off_b, off_c+1): 1},
            {(off_a+1, off_c+1): 1}
        )
        
        # M7: A=(A01-A11), B=(B10+B11)
        # C contributions: C00 += 1*M7
        add_term(
            {(off_a, off_b+1): 1, (off_a+1, off_b+1): -1},
            {(off_b+1, off_c): 1, (off_b+1, off_c+1): 1},
            {(off_a, off_c): 1}
        )
    
    # Schoolbook for mixed-size blocks.
    # For A block of size (ra, ca) and B block of size (rb, cb) where ca == rb,
    # the C block has size (ra, cb).
    # Schoolbook: for each (i,j) in the C block, sum over the common dimension.
    # We'll add one term per (row, col) in the C block.
    
    def schoolbook(a, b, c):
        """Add schoolbook terms for block multiplication A[a][:,b] * B[b][:,c]."""
        off_a, off_b, off_c = offsets[a], offsets[b], offsets[c]
        sa, sb, sc = sizes[a], sizes[b], sizes[c]
        # A block: rows off_a..off_a+sa-1, cols off_b..off_b+sb-1
        # B block: rows off_b..off_b+sb-1, cols off_c..off_c+sc-1
        # C block: rows off_a..off_a+sa-1, cols off_c..off_c+sc-1
        for i in range(sa):
            for k in range(sc):
                u = {}
                v = {}
                w = {}
                for j in range(sb):
                    u[(off_a+i, off_b+j)] = 1
                    v[(off_b+j, off_c+k)] = 1
                w[(off_a+i, off_c+k)] = 1
                add_term(u, v, w)
    
    # Now, identify which block multiplications to use Strassen for.
    # Strassen applies when all three blocks are 2x2.
    # So a, b, c must all be in {0, 1} (the first two blocks).
    
    # For C11 (a=0, c=0): sum over b of A[0,b]*B[b,0]
    #   b=0: 2x2 * 2x2 -> Strassen
    #   b=1: 2x2 * 2x2 -> Strassen
    #   b=2: 2x1 * 1x2 -> schoolbook
    # For C12 (a=0, c=1): 
    #   b=0: 2x2 * 2x2 -> Strassen
    #   b=1: 2x2 * 2x2 -> Strassen
    #   b=2: 2x1 * 1x2 -> schoolbook
    # etc.
    
    # But Strassen computes all 4 C blocks at once. So if we call strassen_2x2(0, 0, 0),
    # it adds terms for C00, C01, C10, C11 (in block coordinates).
    # We need to be careful not to double-count.
    
    # Let me restructure: for each pair (a, c) where a, c are block indices,
    # and for each b, we need to compute A[a,b] * B[b,c].
    # If sizes[a] == sizes[b] == sizes[c] == 2, use Strassen.
    # Otherwise, use schoolbook.
    
    # But Strassen computes all 4 output blocks. So we should call it once per (a,b) pair
    # where a, b are 2x2 blocks, and it will handle all c.
    
    # Actually, Strassen(2x2 A, 2x2 B) gives 2x2 C. So for fixed a, b (both 2x2),
    # Strassen gives C[a,b] for all c where B[b,c] is 2x2.
    # Hmm, this is getting complicated. Let me just enumerate all (a, b, c) triples
    # and decide for each whether to use Strassen or schoolbook.
    # But Strassen computes 4 C blocks at once, so I should group by (a, b) where both are 2x2.
    
    # Let's do it this way:
    # For each (a, b) where sizes[a] == 2 and sizes[b] == 2:
    #   For each c where sizes[c] == 2:
    #     Use Strassen to compute C[a,c] from A[a,b] and B[b,c].
    #   But Strassen computes all 4 C blocks at once, so I should call it once per (a,b) pair
    #   and it will cover all c.
    
    # Actually, the cleanest approach: call strassen_2x2(a, b, c) for each (a, b, c)
    # where all three are 2x2 blocks. But this will add redundant terms because
    # strassen_2x2(0, 0, 0) and strassen_2x2(0, 0, 1) both contribute to the same
    # A and B entries.
    
    # I think the issue is that Strassen is for a single 2x2 * 2x2 = 2x2 multiplication.
    # So strassen_2x2(a, b, c) computes C[a,c] = A[a,b] * B[b,c] for specific a, b, c.
    # Each call is independent and correct.
    # The 7 terms are specific to that (a, b, c) triple.
    
    # So I should call strassen_2x2(a, b, c) for each (a, b, c) where all sizes are 2,
    # and schoolbook(a, b, c) for the rest.
    
    for a in range(nb):
        for b in range(nb):
            for c in range(nb):
                if sizes[a] == 2 and sizes[b] == 2 and sizes[c] == 2:
                    strassen_2x2(a, b, c)
                else:
                    schoolbook(a, b, c)
    
    # Now build U, V, W from the terms.
    # U index: i*m + j for i in 0..n-1, j in 0..m-1
    # V index: j*p + k for j in 0..m-1, k in 0..p-1
    # W index: k*n + i for k in 0..p-1, i in 0..n-1
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[i * m + j][r] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][r] = val
        for (i, k), val in w_dict.items():
            W[k * n + i][r] = val
    
    # Verify the decomposition
    # For each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    # For other (i, j, k) combinations that aren't valid A*B entries, it can be anything
    # (the tensor only has non-zero entries at valid positions).
    # Actually, the tensor M_{n,m,p} has M[k*n+i, i*m+j, j*p+k] = 1 for all i,j,k.
    # So we need the sum to be 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
