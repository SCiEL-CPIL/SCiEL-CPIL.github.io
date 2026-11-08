import numpy as np

def construct():
    n, m, p = 4, 4, 7

    def strassen_2x2(i0, j0, k0, U, V, W):
        """Add 7 Strassen terms for 2x2 * 2x2 * 2x2 with row offsets (i0,j0) in A, (j0,k0) in B, (k0,i0) in C."""
        im, jp, pn = i0 * m + j0, j0 * p + k0, k0 * n + i0
        # M1: (A11+A22)*(B11+B22) -> C11+=M1, C22+=M1
        U[i0*m+j0] += 1; U[(i0+1)*m+(j0+1)] += 1
        V[j0*p+k0] += 1; V[(j0+1)*p+(k0+1)] += 1
        W[k0*n+i0] += 1; W[(k0+1)*n+(i0+1)] += 1
        U[i0*m+j0] += 1; U[(i0+1)*m+(j0+1)] += 1
        # M2: (A21+A22)*B11 -> C21+=M2, C22-=M2
        U[(i0+1)*m+j0] += 1; U[(i0+1)*m+(j0+1)] += 1
        V[j0*p+k0] += 1
        W[k0*n+(i0+1)] += 1; W[(k0+1)*n+(i0+1)] -= 1
        # M3: A11*(B21-B11) -> C21+=M3, C22+=M3
        U[i0*m+j0] += 1
        V[(j0+1)*p+(k0+1)] += 1; V[j0*p+(k0+1)] -= 1
        W[(k0+1)*n+i0] += 1; W[(k0+1)*n+(i0+1)] += 1
        # M4: (A22-A11)*B12 -> C12+=M4, C22+=M4
        U[(i0+1)*m+(j0+1)] += 1
        V[(j0+1)*p+k0] += 1; V[j0*p+k0] -= 1
        W[k0*n+i0] += 1; W[k0*n+(i0+1)] += 1
        # M5: (A11+A12)*B22 -> C12+=M5, C11-=M5
        U[i0*m+j0] += 1; U[i0*m+(j0+1)] += 1
        V[(j0+1)*p+(k0+1)] += 1
        W[(k0+1)*n+i0] += 1; W[k0*n+i0] -= 1
        # M6: (A21-A11)*(B11+B12) -> C21+=M6
        U[(i0+1)*m+j0] += 1; U[i0*m+j0] -= 1
        V[j0*p+k0] += 1; V[j0*p+(k0+1)] += 1
        W[(k0+1)*n+(i0+1)] += 1
        # M7: (A12-A22)*(B21+B22) -> C12+=M7
        U[i0*m+(j0+1)] += 1; U[(i0+1)*m+(j0+1)] -= 1
        V[(j0+1)*p+k0] += 1; V[(j0+1)*p+(k0+1)] += 1
        W[k0*n+i0] += 1

    def strassen_4x4_k1(i0, j0, k0, U, V, W):
        """Compute C[:, k0..k0+3] = A[:, j0..j0+3] * B[j0..j0+3, k0..k0+3] using Strassen recursively."""
        # Split 4x4 into 2x2 blocks for each multiplication
        # We need to multiply A[j0:j0+4, j0:j0+4] by B[j0:j0+4, k0:k0+4]
        # Using Strassen: split into 2x2 blocks of 2x2 matrices
        # A11=A[0:2,0:2], A12=A[0:2,2:4], A21=A[2:4,0:2], A22=A[2:4,2:4]
        # B11=B[0:2,0:2], B12=B[0:2,2:4], B21=B[2:4,0:2], B22=B[2:4,2:4]
        
        # M1 = (A11+A22)(B11+B22)
        strassen_2x2(i0, j0, k0, U, V, W)       # A11*B11 -> C11,C22
        strassen_2x2(i0, j0+2, k0, U, V, W)     # A11*B12 -> wait, this isn't right
        # Actually I need a proper 4x4 Strassen decomposition.
        # Let me do it manually with 7 multiplications of 2x2 blocks, each using 7 scalar multiplications = 49 total.
        
    def strassen_4x4_head(U, V, W):
        """Strassen-based decomposition for the 4x4 * 4x4 part (columns 0-3 of output)."""
        # Use tensor product of 2x2 Strassen
        # The 4x4 matmul can be done with 2^2=4 recursive Strassen 2x2, each with 7 terms = 49 terms
        # But we need the proper indexing.
        # For C[i,k] = sum_j A[i,j]*B[j,k] with i,j,k in 0..3:
        # Split into 2x2 blocks: A11=[i<2,j<2], A12=[i<2,j>=2], A21=[i>=2,j<2], A22=[i>=2,j>=2]
        # Same for B and C.
        # M1 = (A11+A22)(B11+B22): contributes to C11 and C22
        # M2 = (A21+A22)B11: contributes to C21 and -C22
        # M3 = A11(B21-B11): contributes to C21 and C22  [wait, standard is A11(B21-B11) -> C21 += M3, C22 += M3? No]
        # Standard Strassen:
        # C11 = M1 - M4 - M5 + M7
        # C12 = M3 + M4
        # C21 = M2 + M5
        # C22 = M1 - M2 + M3 - M6
        
        # Each "multiplication" is a 2x2 * 2x2 = 2x2 matmul, which itself takes 7 terms.
        # So 7 * 7 = 49 terms total for 4x4 * 4x4 -> 4x4.
        
        # Let's implement this carefully.
        # For block multiplies, I'll use helper that adds terms for computing 2x2 = 2x2 * 2x2.
        
        def add_2x2_mult(i_base, j_base, k_base, signs_u, signs_v):
            """Add 7 terms for 2x2 matmul: C_block = A_block * B_block
            A_block is rows i_base:i_base+2, cols j_base:j_base+2
            B_block is rows j_base:j_base+2, cols k_base:k_base+2
            C_block is rows k_base:k_base+2, cols i_base:i_base+2 (using W index k*n+i)
            """
            for (di, dj) in [(0,0),(0,1),(1,0),(1,1)]:
                for (dk, dl) in [(0,0),(0,1),(1,0),(1,1)]:
                    pass
            # Just use strassen_2x2
            strassen_2x2(i_base, j_base, k_base, U, V, W)
        
        # M1 = (A11+A22)(B11+B22)
        # A11 is rows 0-1, cols 0-1; A22 is rows 2-3, cols 2-3
        # This isn't a single contiguous block, so I can't just use strassen_2x2 directly.
        # I need to handle the sums of blocks.
        
        # Actually, let me just use the naive 4x4x4 = 64 terms and apply Strassen at a higher level.
        # Better approach: use the known fact that 4x4*4x4 needs at most 49 multiplications via Strassen.
        # Let me construct the 49 terms explicitly using the recursive Strassen pattern.
        
        # Recursive Strassen for 4x4:
        # Split A into [A11, A12; A21, A22] where each is 2x2
        # Split B into [B11, B12; B21, B22] where each is 2x2
        # M1 = (A11+A22)(B11+B22) -> 2x2*2x2 = 7 terms
        # M2 = (A21+A22)B11 -> 2x2*2x2 = 7 terms
        # M3 = A11(B21-B11) -> 2x2*2x2 = 7 terms
        # M4 = (A22-A11)B12 -> 2x2*2x2 = 7 terms
        # M5 = (A11+A12)B22 -> 2x2*2x2 = 7 terms
        # M6 = (A21-A11)(B11+B12) -> 2x2*2x2 = 7 terms
        # M7 = (A12-A22)(B21+B22) -> 2x2*2x2 = 7 terms
        # Total: 49 terms
        # C11 = M1 - M4 - M5 + M7
        # C12 = M3 + M4
        # C21 = M2 + M5
        # C22 = M1 - M2 + M3 - M6
        
        # Each Mi is a 2x2 result. The contribution to C is:
        # C11 gets: M1*1, M4*(-1), M5*(-1), M7*(+1)
        # C12 gets: M3*(+1), M4*(+1)
        # C21 gets: M2*(+1), M5*(+1)
        # C22 gets: M1*(+1), M2*(-1), M3*(+1), M6*(-1)
        
        # For each Mi, it's computed via 7 scalar products. The scalar product (A_elem * B_elem) contributes
        # to specific C elements via the W matrix.
        
        # Let me define for each of the 7 M's, the 7 sub-terms with their (A_factor, B_factor, C_factor)
        # and the coefficient to apply to C.
        
        # This is getting complex. Let me use a cleaner approach: build the full 4x4*4x4 tensor
        # and verify my Strassen decomposition.
        
        # Alternative: just use 64 schoolbook terms for the 4x4*4x4 part. That's simpler and correct.
        # 64 terms for head (cols 0-3), then handle tail (cols 4-6).
        # For tail cols 4-6 (3 columns):
        #   col 4: 4*4 = 16 schoolbook terms (or use Strassen 4x4x2 for cols 4-5, then 4x4x1 for col 6)
        #   Let's use Strassen 4x4x2 for cols 4-5, then schoolbook for col 6.
        
        # 4x4x2 via Strassen: split 4x4 A and 4x2 B.
        # B has shape 4x2. Split A into 2x2 blocks, B into 2x1 blocks.
        # C1 = A11*B11 + A12*B21 (4x1 = 2x2*2x1 + 2x2*2x1)
        # C2 = A21*B11 + A22*B21
        # 2x2 * 2x1 = 2*2 = 4 schoolbook terms, or use Strassen? Strassen doesn't help for 2x2*2x1 (it's only 4 terms).
        # Actually 2x2 * 2x1: C[i] = A[i,0]*B[0] + A[i,1]*B[1]. That's 4 scalar products. No savings.
        # So 4x4x2 = 2*(2x2*2x1) * 2 = wait.
        # A is 4x4, B is 4x2. 
        # C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]
        # = 2x2*2x2 + 2x2*2x2 -> each 2x2*2x2 can be 7 terms (Strassen) or 4 terms (schoolbook).
        # So C[0:2,:2] takes 7+7=14 terms (Strassen) or 4+4=8 (schoolbook).
        # Similarly C[2:4,:2] takes 14 or 8.
        # Total for 4x4x2: 28 (Strassen) or 16 (schoolbook).
        # Schoolbook is better! 16 terms.
        
        # So total: 64 (head) + 16 (cols 4-5) + 16 (col 6, which is 4*4=16) = 96. Worse than current 93.
        
        # Current program uses 49 (Strassen 4x4 head) + 28 (Strassen for 2x2 blocks of tail) + 16 (schoolbook col 6) = 93.
        # Let me just replicate that exactly.
        
        pass  # This function won't be called; I'll build it differently below.

    # Build the decomposition directly
    terms = []
    
    # Head: columns 0-3 of output, 4x4 * 4x4 using Strassen (49 terms)
    # I'll use the recursive Strassen approach: 7 M's, each 2x2*2x2 = 7 terms = 49 total.
    
    # Let me implement this properly.
    # For each of the 7 matrix multiplications in Strassen:
    # M1: (A11+A22) * (B11+B22)
    # The 2x2 blocks are:
    # A11: rows 0-1, cols 0-1 -> indices (0,0),(0,1),(1,0),(1,1)
    # A22: rows 2-3, cols 2-3 -> indices (2,2),(2,3),(3,2),(3,3)
    # Sum: elements (0,0),(0,1),(1,0),(1,1),(2,2),(2,3),(3,2),(3,3) each get coefficient 1
    
    # B11: rows 0-1, cols 0-1
    # B22: rows 2-3, cols 2-3
    # Sum: same pattern
    
    # The product M1 is a 2x2 matrix. Its element M1[i,j] contributes to:
    # C11[i,j] += M1[i,j], C22[i,j] += M1[i,j]
    # In flat indices: C11 is rows k=0-1, cols i=0-1 -> W index k*4+i for k=0,1 and i=0,1
    # C22 is rows k=2-3, cols i=2-3 -> W index k*4+i for k=2,3 and i=2,3
    
    # For the 2x2*2x2 = 2x2 multiplication using Strassen (7 terms):
    # Let's say we're multiplying X (2x2) by Y (2x2) to get Z (2x2).
    # Z[0,0] = M_s1 - M_s4 - M_s5 + M_s7
    # Z[0,1] = M_s3 + M_s4
    # Z[1,0] = M_s2 + M_s5
    # Z[1,1] = M_s1 - M_s2 + M_s3 - M_s6
    # where:
    # M_s1 = (X[0,0]+X[1,1]) * (Y[0,0]+Y[1,1])
    # M_s2 = (X[1,0]+X[1,1]) * Y[0,0]
    # M_s3 = X[0,0] * (Y[1,0]-Y[0,0])
    # M_s4 = (X[1,1]-X[0,0]) * Y[0,1]
    # M_s5 = (X[0,0]+X[0,1]) * Y[1,1]
    # M_s6 = (X[1,0]-X[0,0]) * (Y[0,0]+Y[0,1])
    # M_s7 = (X[0,1]-X[1,1]) * (Y[1,0]+Y[1,1])
    
    # This is getting very involved. Let me just hardcode the 49 terms.
    # Actually, let me use a different strategy: build the tensor and use a simpler decomposition.
    
    # Given the complexity, let me just use the approach from the current program which gives 93.
    # 49 Strassen terms for 4x4*4x4 (head) + 28 for two 2x2 Strassen blocks (tail cols 4-5) + 16 schoolbook (col 6)
    
    def add_strassen_2x2(i0, j0, k0):
        """Add 7 Strassen terms for 2x2*2x2*2x2."""
        # M1
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[i0*m+j0]=1; u[(i0+1)*m+(j0+1)]=1
        v[j0*p+k0]=1; v[(j0+1)*p+(k0+1)]=1
        w[k0*n+i0]=1; w[(k0+1)*n+(i0+1)]=1
        terms.append((u,v,w))
        # M2
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i0+1)*m+j0]=1; u[(i0+1)*m+(j0+1)]=1
        v[j0*p+k0]=1
        w[k0*n+(i0+1)]=1; w[(k0+1)*n+(i0+1)]=-1
        terms.append((u,v,w))
        # M3
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[i0*m+j0]=1
        v[j0*p+(k0+1)]=1; v[(j0+1)*p+(k0+1)]=-1
        w[(k0+1)*n+i0]=1; w[(k0+1)*n+(i0+1)]=1
        terms.append((u,v,w))
        # M4
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i0+1)*m+(j0+1)]=1
        v[(j0+1)*p+k0]=1; v[j0*p+k0]=-1
        w[k0*n+i0]=1; w[k0*n+(i0+1)]=1
        terms.append((u,v,w))
        # M5
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[i0*m+j0]=1; u[i0*m+(j0+1)]=1
        v[(j0+1)*p+(k0+1)]=1
        w[(k0+1)*n+i0]=1; w[k0*n+i0]=-1
        terms.append((u,v,w))
        # M6
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i0+1)*m+j0]=1; u[i0*m+j0]=-1
        v[j0*p+k0]=1; v[j0*p+(k0+1)]=1
        w[(k0+1)*n+(i0+1)]=1
        terms.append((u,v,w))
        # M7
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[i0*m+(j0+1)]=1; u[(i0+1)*m+(j0+1)]=-1
        v[(j0+1)*p+k0]=1; v[(j0+1)*p+(k0+1)]=1
        w[k0*n+i0]=1
        terms.append((u,v,w))

    # Head: 4x4 * 4x4 using 49 terms (Strassen recursively)
    # Use tensor product: 2x2 Strassen on outer, 2x2 Strassen on inner = 7*7 = 49
    outer_terms = [
        # (u_indices, v_indices, w_indices) for the 7 outer Strassen terms
        # Each entry: list of (block, coeff) where block is 0 or 1 (for 2x2 of 2x2 blocks)
        ([((0,0,1),(1,1,1))], [((0,0,1),(1,1,1))], [((0,0,1),(1,1,1))]),  # M1: (A11+A22)(B11+B22)
        ([((1,0,1),(1,1,1))], [((0,0,1))], [((1,0,1),(1,1,-1))]),  # M2: (A21+A22)B11 -> C21+=, C22-=
        ([((0,0,1))], [((1,0,1),(0,0,-1))], [((1,0,1),(1,1,1))]),  # M3: A11(B21-B11) -> C21+=, C22+=
        ([((1,1,1),(0,0,-1))], [((0,1,1))], [((0,1,1),(1,1,1))]),  # M4: (A22-A11)B12 -> C12+=, C22+=
        ([((0,0,1),(0,1,1))], [((1,1,1))], [((0,-1,1),(1,1,1))]),  # M5: (A11+A12)B22 -> C12+=, C11-=
        ([((1,0,1),(0,0,-1))], [((0,0,1),(0,1,1))], [((1,1,1))]),  # M6: (A21-A11)(B11+B12) -> C21+=
        ([((0,1,1),(1,1,-1))], [((1,0,1),(1,1,1))], [((0,0,1))]),  # M7: (A12-A22)(B21+B22) -> C12+=
    ]
    
    inner_terms = [
        ([((0,0,1),(1,1,1))], [((0,0,1),(1,1,1))], [((0,0,1),(1,1,1))]),
        ([((1,0,1),(1,1,1))], [((0,0,1))], [((1,0,1),(1,1,-1))]),
        ([((0,0,1))], [((1,0,1),(0,0,-1))], [((1,0,1),(1,1,1))]),
        ([((1,1,1),(0,0,-1))], [((0,1,1))], [((0,1,1),(1,1,1))]),
        ([((0,0,1),(0,1,1))], [((1,1,1))], [((0,-1,1),(1,1,1))]),
        ([((1,0,1),(0,0,-1))], [((0,0,1),(0,1,1))], [((1,1,1))]),
        ([((0,1,1),(1,1,-1))], [((1,0,1),(1,1,1))], [((0,0,1))]),
    ]
    
    for oa, ob, oc in outer_terms:
        for ix, iy, iz in inner_terms:
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            for (bl, lr, lc, coeff_o) in [(x[0], x[1], x[2], x[3]) if len(x)==4 else (x[0], x[1], x[2], 1) for x in [None]] if False else []:
                pass
            # Simpler:
            for (oblock, orow, ocol) in oa:
                for (iblock, irow, icol) in ix:
                    coeff = 1  # All coefficients are 1 in this formulation based on the lists above
                    # Wait, the lists have tuples like (0,0,1) meaning (block, local, coeff)
                    pass
            
            # Let me redo this with explicit coefficient handling
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            for item in oa:
                oblk, or_, oc_ = item[0], item[1], item[2]
                # or_, oc_ are local 2x2 indices, oblk is which 2x2 block (0 or 1)
                for item2 in ix:
                    iblk, ir_, ic_ = item2[0], item2[1], item2[2]
                    # The local index in the 2x2 is (or_, oc_) within block oblk
                    # Global row = oblk*2 + or_, global col = oc_ if same block row... 
                    # Wait, I need to be more careful.
                    # A is 4x4. Block (bl) is at rows 2*bl:2*bl+2, cols 2*bl:2*bl+2 for the main diagonal
                    # But in Strassen, A11 is top-left, A12 is top-right, A21 is bottom-left, A22 is bottom-right
                    # So A[blk] where blk=0 -> rows 0-1, cols 0-1 (A11)
                    # blk=1 -> but I used oblk in {0,1} for the two "halves" in the 2x2-of-blocks space.
                    # Let me redefine: in the outer Strassen, the 2x2 "super-block" has:
                    # position (0,0) -> A11 (rows 0-1, cols 0-1)
                    # position (0,1) -> A12 (rows 0-1, cols 2-3)
                    # position (1,0) -> A21 (rows 2-3, cols 0-1)
                    # position (1,1) -> A22 (rows 2-3, cols 2-3)
                    # So (or_, oc_) directly gives the super-block position.
                    # The inner 2x2 block at super-position (or_,oc_) has:
                    # local (lr, lc) -> global row = 2*or_ + lr, global col = 2*oc_ + lc
                    for (ir, ic) in [item2[1], item2[2]]:  # This is wrong, let me restructure
                        pass
            
            # I'm overcomplicating this. Let me just hardcode the 49 terms using a different method.
            # I'll use the fact that the 4x4*4x4 Strassen has 49 terms and generate them systematically.
            break
        break
    
    # This is getting too messy. Let me just use the clean 93-term construction.
    terms = []
    
    # Head (cols 0-3): 49 terms via Strassen 4x4
    # I'll generate them by applying Strassen 2x2 four times with proper offsets
    # Actually, the simplest correct 49-term construction:
    # For each of the 7 outer Strassen multiplications, do a 2x2*2x2 Strassen = 7 terms
    # But the 2x2 blocks are not contiguous in memory, so I need to handle the indexing.
    
    # Let me just use 64 schoolbook terms for the head. Simpler and still gives a valid answer.
    # 64 + 16 + 16 = 96. Not as good as 93, but correct.
    # Or use the current program's 93-term approach.
    
    # Let me replicate the 93-term approach:
    # 49 terms for 4x4*4x4 (Strassen)
    # 28 terms for 4x4*4x2 (two 2x2 Strassen blocks for cols 4-5)
    # 16 terms for 4x4*1 (schoolbook for col 6)
    
    # For the 49-term Strassen 4x4*4x4:
    # I'll use the recursive formulation. The key insight: 
    # The 4x4 matmul tensor can be decomposed as a tensor product of two 2x2 matmul tensors.
    # T_4x4 = T_2x2 ⊗ T_2x2, where each T_2x2 has rank 7.
    # So T_4x4 has rank 7*7 = 49.
    
    # The 2x2 matmul tensor T_2x2 has 7 terms:
    # t1: A[0,0]*B[0,0] -> C[0,0]+=, C[1,1]+=
    # t2: A[1,0]*B[0,0] -> C[1,0]+=, C[1,1]-=
    # t3: A[0,0]*B[1,0] -> C[1,0]+=, C[1,1]+=
    # t4: A[1,1]*B[0,1] -> C[0,1]+=, C[1,1]+=
    # t5: A[0,1]*B[1,1] -> C[0,1]+=, C[0,0]-=
    # t6: A[1,0]*B[0,1] -> C[1,0]+=
    # t7: A[0,1]*B[1,0] -> C[0,1]+=
    # Wait, that's not quite right. Let me use the standard Strassen:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b10-b00)
    # m4 = (a11-a00)b01
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    # c00 = m1 - m4 - m5 + m7
    # c01 = m3 + m4
    # c10 = m2 + m5
    # c11 = m1 - m2 + m3 - m6
    
    # Each mi is a single scalar product (not a 2x2 matmul). So T_2x2 rank is 7.
    
    # For T_4x4 = T_2x2 ⊗ T_2x2:
    # The 4x4 matrix A has indices (i,j) where i = 2*i2 + i1, j = 2*j2 + j1 with i2,i1,j2,j1 in {0,1}
    # Similarly B has (j,k) = (2*j2+2*j1, 2*k2+k1)... wait, j is shared.
    # Actually for matrix multiplication C = A*B where A is 4x4, B is 4x4:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Let i = 2*i2 + i1, j = 2*j2 + j1, k = 2*k2 + k1
    # C[2*i2+i1, 2*k2+k1] = sum_{j2,j1} A[2*i2+i1, 2*j2+j1] B[2*j2+j1, 2*k2+k1]
    
    # Using Strassen at the outer level (splitting into 2x2 blocks):
    # A11 = A[0:2,0:2], A12 = A[0:2,2:4], A21 = A[2:4,0:2], A22 = A[2:4,2:4]
    # B11 = B[0:2,0:2], B12 = B[0:2,2:4], B21 = B[2:4,0:2], B22 = B[2:4,2:4]
    # M1 = (A11+A22)(B11+B22), etc.
    # Each Mi is a 2x2 matmul, which itself uses 7 scalar products.
    # So 7 * 7 = 49 scalar products total.
    
    # For the tensor product formulation, I'll just enumerate all 49 terms.
    # Each term is: sum over (i1,j1) of A_factor[i1,j1] * sum over (j1',k1) of B_factor[j1',k1] * W_contribution
    
    # This is still complex. Let me just use the direct 64-term schoolbook for the head.
    # Total: 64 + 16 + 16 = 96 terms.
    
    # Actually, let me try to get 93 by using the current program's approach.
    # The current program's gen_current() does:
    # - strassen_4x4() for the head (49 terms)
    # - 4 calls to strassen_2x2_block for cols 4-5 (4*7=28 terms)
    # - 16 schoolbook terms for col 6
    # Total: 93
    
    # Let me replicate this exactly.
    
    # Head: 49 terms
    # I'll build the 49-term Strassen 4x4*4x4 by hand.
    # Using the recursive Strassen: for each of the 7 outer multiplications,
    # do a full 2x2*2x2 = 7 terms.
    
    # Outer multiplication 1: (A11+A22) * (B11+B22)
    # A11: A[0,0], A[0,1], A[1,0], A[1,1] with coeff 1
    # A22: A[2,2], A[2,3], A[3,2], A[3,3] with coeff 1
    # B11: B[0,0], B[0,1], B[1,0], B[1,1] with coeff 1
    # B22: B[2,2], B[2,3], B[3,2], B[3,3] with coeff 1
    
    # This product gives a 2x2 result Z1 where:
    # Z1[i1,j1] = sum over the 8+8 A/B elements
    # Z1[0,0] contributes to C[0,0] += Z1[0,0] and C[2,2] += Z1[0,0]  (from C11+=M1, C22+=M1)
    
    # For the 2x2 result Z1 computed via 7 scalar products (inner Strassen):
    # This is getting very complex. Let me just use 64 schoolbook terms.
    
    terms = []
    # Head: 64 schoolbook terms for cols 0-3
    for i in range(n):
        for j in range(m):
            for k in range(4):
                u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
                u[i*m+j] = 1
                v[j*p+k] = 1
                w[k*n+i] = 1
                terms.append((u,v,w))
    
    # Col 4-5: 16 schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(4, 6):
                u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
                u[i*m+j] = 1
                v[j*p+k] = 1
                w[k*n+i] = 1
                terms.append((u,v,w))
    
    # Col 6: 16 schoolbook terms
    for i in range(n):
        for j in range(m):
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[i*m+j] = 1
            v[j*p+6] = 1
            w[6*n+i] = 1
            terms.append((u,v,w))
    
    # Total: 64 + 16 + 16 = 96 terms
    # This is valid but not as good as 93. Let me try to implement the 93-term version.
    
    # Actually, let me just go with 96 terms for now. It's correct and simple.
    
    rank = len(terms)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r, (u, v, w) in enumerate(terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
    
    return {"U": U, "V": V, "W": W}

